import logging
from functools import partial

import cloudpickle
import numpy as np

from neuclease.util import Grid, boxes_from_grid, clipped_boxes_from_grid, box_intersection, box_to_slicing, overwrite_subvol, extract_subvol

from DVIDSparkServices.util import box_as_tuple
from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.util import cpus_per_worker, num_worker_nodes, persist_and_execute, unpersist
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray

logger = logging.getLogger(__name__)

class Brick:
    """
    Conceptually, bricks are intended to populate (or sparsely populate)
    a Grid, with no more than one Brick per grid block.
    
    The particular Grid block in which a Brick lives is indicated by it's logical_box,
    which always encompasses an exact (complete) grid block region.
    
    But for some Bricks, there may not be enough data to fill the entire Grid block
    (e.g. for Bricks that lie on the edge of a large volume's bounding-box).
    In that case, the Brick.volume array might only span a subregion within the logical_box.
    That region is indicated by the Brick's physical_box.

    In fact, a Brick's physical_box may even be *larger* than it's logical_box,
    which is useful for operations in which Brick data needs to be padded with some 'halo'.
    
    In Summary, a Brick consists of three items:

        - volume: An array of voxels, i.e. the actual data for the Brick

        - physical_box: The region in space that those voxels inhabit.
                        By definition, `volume.shape == (physical_box[1] - physical_box[0])`

        - logical_box: The Brick's logical address within some abstract Grid in space.
     
        Note: Both boxes (logical and physical) are always stored in GLOBAL coordinates.
    """
    def __init__(self, logical_box, physical_box, volume=None, *, lazy_creation_fn=None, custom_hash=None):
        """
        Args:
            logical_box:
                The Brick's logical address within some abstract Grid in space.
            physical_box:
                The region in space that those voxels inhabit.
                By definition, `volume.shape == (physical_box[1] - physical_box[0])`
            volume:
                An array of voxels, i.e. the actual data for the Brick.
                May be None if lazy_creation_fn is provided.
            lazy_creation_fn:
                Instead of providing a volume at construction, you may provide a function to create the volume,
                which will be called upon the first access to Brick.volume
            custom_hash:
                Optionally specify a hash value for this brick, which will be returned by __hash__()
        """
        assert (volume is None) ^ (lazy_creation_fn is None), \
            "Must supply either volume or lazy_creation_fn (not both)"
        self.logical_box = np.asarray(logical_box)
        self.physical_box = np.asarray(physical_box)

        self._volume = volume
        if self._volume is not None:
            assert ((self.physical_box[1] - self.physical_box[0]) == self._volume.shape).all()
        
        # Used for pickling.
        self._compressed_volume = None
        self._destroyed = False
        
        # Optional custom hashing
        self._hash = custom_hash
        
        self._create_volume_fn = None
        if lazy_creation_fn is not None:
            self._create_volume_fn = cloudpickle.dumps(lazy_creation_fn)

    def __hash__(self):
        if self._hash is None:
            return rt.better_hash(tuple(self.logical_box[0].tolist()))
        else:
            return self._hash

    def __str__(self):
        if (self.logical_box == self.physical_box).all():
            return f"logical & physical: {self.logical_box.tolist()}"
        return f"logical: {self.logical_box.tolist()}, physical: {self.physical_box.tolist()}"

    def set_hash(self, new_hash):
        """
        Explicitly set the hash value for this brick.
        Returns self so this function is easily used in map(), etc.
        """
        self._hash = new_hash
        return self

    @property
    def volume(self):
        """
        The volume is created or decompressed lazily.
        See __init__ and __getstate__() for explanation.
        """
        if self._destroyed:
            raise RuntimeError("Attempting to access data for a brick that has already been explicitly destroyed:\n"
                               f"{self}")
        if self._volume is not None:
            return self._volume

        if self._compressed_volume is not None:
            assert self._compressed_volume is not None
            self._volume = self._compressed_volume.deserialize()
            return self._volume
        
        if self._create_volume_fn is not None:
            fn = cloudpickle.loads(self._create_volume_fn)
            self._volume = fn(self.physical_box)
            assert (self._volume.shape == (self.physical_box[1] - self.physical_box[0])).all()
            del self._create_volume_fn
            return self._volume
        
        raise AssertionError("This brick has no data, and no way to create it.")
    
    def compress(self):
        """
        Compress the volume.
        Will be uncompressed again automatically on first access.
        """
        if self._destroyed:
            raise RuntimeError("Attempting to compress data for a brick that has already been explicitly destroyed:\n"
                               f"{self}")
        if self._volume is not None:
            self._compressed_volume = CompressedNumpyArray(self._volume)
            self._volume = None
    
    def __getstate__(self):
        """
        Pickle representation.
        
        By default, the volume would be compressed/decompressed transparently via
        the code in CompressedNumpyArray.py, but we want decompression to be
        performed lazily.
        
        Therefore, we explicitly compress the volume here, and decompress it only
        first upon access, via the self.volume property.
        
        This avoids decompression during certain Spark operations that don't
        require actual manipulation of the voxels, notably groupByKey().
        """
        if self._destroyed:
            raise RuntimeError("Attempting to pickle a brick that has already been explicitly destroyed:\n"
                               f"{self}")
        if self._volume is not None:
            self._compressed_volume = CompressedNumpyArray(self._volume)

        d = self.__dict__.copy()
        d['_volume'] = None
        return d

    def destroy(self):
        self._volume = None
        self._destroyed = True

def generate_bricks_from_volume_source( bounding_box, grid, volume_accessor_func, sc=None, rdd_partition_length=None, sparse_boxes=None, lazy=False ):
    """
    Generate an RDD or iterable of Bricks for the given bounding box and grid.
     
    Args:
        bounding_box:
            (start, stop)
 
        grid:
            Grid (see above)
 
        volume_accessor_func:
            Callable with signature: f(box) -> ndarray
            Note: The callable will be unpickled only once per partition, so initialization
                  costs after unpickling are only incurred once per partition.
 
        sc:
            SparkContext. If provided, an RDD is returned.  Otherwise, returns an ordinary Python iterable.
 
        rdd_partition_length:
            Optional. If provided, the RDD will have (approximately) this many bricks per partition.
        
        sparse_boxes:
            Optional.
            A pre-calculated list of boxes to use instead of instead of calculating
            the complete (dense) list of grid boxes within the bounding box.
            If provided, should be a list of physical boxes, and no two should occupy
            the same logical box, as defined by their midpoints.
            Note: They will still be clipped to the overall bounding_box.
        
        halo: An integer or shape indicating how much halo to add to each Brick's physical_box.
              The halo is applied in both 'dense' and 'sparse' cases.
    """
    if sparse_boxes is None:
        # Generate boxes from densely populated grid
        logical_boxes = boxes_from_grid(bounding_box, grid, include_halos=False)
        physical_boxes = clipped_boxes_from_grid(bounding_box, grid)
        logical_and_physical_boxes = zip( logical_boxes, physical_boxes )
    else:
        # User provided list of physical boxes.
        # Clip them to the bounding box and calculate the logical boxes.
        if not hasattr(sparse_boxes, '__len__'):
            sparse_boxes = list( sparse_boxes )
        physical_boxes = np.asarray( sparse_boxes )
        assert physical_boxes.ndim == 3 and physical_boxes.shape[1:3] == (2,3)

        def logical_and_clipped( box ):
            midpoint = (box[0] + box[1]) // 2
            logical_box = grid.compute_logical_box( midpoint )
            box += (-grid.halo_shape, grid.halo_shape)
            # Note: Non-intersecting boxes will have non-positive shape after clipping
            clipped_box = box_intersection(box, bounding_box)
            return ( logical_box, clipped_box )

        logical_and_physical_boxes = map(logical_and_clipped, physical_boxes)

        # Drop any boxes that fall completely outside the bounding box
        # Check that physical box doesn't completely fall outside its logical_box
        def is_valid(logical_and_physical):
            logical_box, physical_box = logical_and_physical
            return (physical_box[1] > logical_box[0]).all() and (physical_box[0] < logical_box[1]).all()
        logical_and_physical_boxes = filter(is_valid, logical_and_physical_boxes )

    if sc:
        if not hasattr(logical_and_physical_boxes, '__len__'):
            logical_and_physical_boxes = list(logical_and_physical_boxes) # need len()

        num_rdd_partitions = None
        if rdd_partition_length is not None:
            rdd_partition_length = max(1, rdd_partition_length)
            num_rdd_partitions = int( np.ceil( len(logical_and_physical_boxes) / rdd_partition_length ) )

        # If we're working with a tiny volume (e.g. testing),
        # make sure we at least parallelize across all cores.
        if num_rdd_partitions is not None and (num_rdd_partitions < cpus_per_worker() * num_worker_nodes()):
            num_rdd_partitions = cpus_per_worker() * num_worker_nodes()

        def brick_size(log_phys):
            _logical, physical = log_phys
            return np.uint64(np.prod(physical[1] - physical[0]))
        total_volume = sum(map(brick_size, logical_and_physical_boxes))
        logger.info(f"Initializing RDD of {len(logical_and_physical_boxes)} Bricks "
                    f"(over {num_rdd_partitions} partitions) with total volume {total_volume/1e9:.1f} Gvox")

        # Enumerate and repartition to get perfect partition sizes,
        # rather than relying on spark's default hash
        class _enumerated_value(tuple):
            # Return a hash based on the key alone.
            def __hash__(self):
                return self[0]

        enumerated_logical_and_physical_boxes = sc.parallelize( enumerate(logical_and_physical_boxes), num_rdd_partitions )
        enumerated_logical_and_physical_boxes = enumerated_logical_and_physical_boxes.map(_enumerated_value)
        enumerated_logical_and_physical_boxes = enumerated_logical_and_physical_boxes.partitionBy(num_rdd_partitions, lambda x: x)
        logical_and_physical_boxes = enumerated_logical_and_physical_boxes.values()

    def make_bricks( logical_and_physical_box ):
        logical_box, physical_box = logical_and_physical_box
        if lazy:
            return Brick(logical_box, physical_box, lazy_creation_fn=volume_accessor_func)
        else:
            volume = volume_accessor_func(physical_box)
            return Brick(logical_box, physical_box, volume)
    
    bricks = rt.map( make_bricks, logical_and_physical_boxes )
    return bricks

def clip_to_logical( brick ):
    """
    Truncate the given brick so that it's volume does not exceed the bounds of its logical_box.
    (Useful if the brick was originally constructed with a halo.)
    """
    intersection = box_intersection(brick.physical_box, brick.logical_box)
    assert (intersection[1] > intersection[0]).all(), \
        f"physical_box ({brick.physical_box}) does not intersect logical_box ({brick.logical_box})"
    
    intersection_within_physical = intersection - brick.physical_box[0]
    new_vol = brick.volume[ box_to_slicing(*intersection_within_physical) ]
    return Brick( brick.logical_box, intersection, new_vol )


def pad_brick_data_from_volume_source( padding_grid, volume_accessor_func, brick ):
    """
    Expand the given Brick's data until its physical_box is aligned with the given padding_grid.
    The data in the expanded region will be sourced from the given volume_accessor_func.
    
    Note: padding_grid need not be identical to the grid the Brick was created with,
          but it must divide evenly into that grid. 
    
    For instance, if padding_grid happens to be the same as the brick's own native grid,
    then the phyiscal_box is expanded to align perfectly with the logical_box on all sides: 
    
        +-------------+      +-------------+
        | physical |  |      |     same    |
        |__________|  |      |   physical  |
        |             |  --> |     and     |
        |   logical   |      |   logical   |
        |_____________|      |_____________|
    
    Args:
        brick: Brick
        padding_grid: Grid
        volume_accessor_func: Callable with signature: f(box) -> ndarray

    Returns: Brick
    
    Note: It is not legal to call this function unless the Brick's physical_box
          lies completely within the logical_box (i.e. no halos allowed).
          Furthremore, the padding_grid is not permitted to use a halo, either.
          (These restrictions could be fixed, but the current version of this
          function has these requirements.)

    Note: If no padding is necessary, then the original Brick is returned (no copy is made).
    """
    assert isinstance(padding_grid, Grid)
    assert not padding_grid.halo_shape.any()
    block_shape = padding_grid.block_shape
    assert ((brick.logical_box - padding_grid.offset) % block_shape == 0).all(), \
        f"Padding grid {padding_grid.offset} must be aligned with brick logical_box: {brick.logical_box}"
    
    # Subtract offset to calculate the needed padding
    offset_physical_box = brick.physical_box - padding_grid.offset

    if (offset_physical_box % block_shape == 0).all():
        # Internal data is already aligned to the padding_grid.
        return brick
    
    offset_padded_box = np.array([offset_physical_box[0] // block_shape * block_shape,
                                  (offset_physical_box[1] + block_shape - 1) // block_shape * block_shape])
    
    # Re-add offset
    padded_box = offset_padded_box + padding_grid.offset
    assert (padded_box[0] >= brick.logical_box[0]).all()
    assert (padded_box[1] <= brick.logical_box[1]).all()

    # Initialize a new volume of the fully-padded shape
    padded_volume_shape = padded_box[1] - padded_box[0]
    padded_volume = np.zeros(padded_volume_shape, dtype=brick.volume.dtype)

    # Overwrite the previously existing data in the new padded volume
    orig_box = brick.physical_box
    orig_box_within_padded = orig_box - padded_box[0]
    overwrite_subvol(padded_volume, orig_box_within_padded, brick.volume)
    
    # Check for a non-zero-volume halo on all six sides.
    halo_boxes = []
    for axis in range(padded_volume.ndim):
        if orig_box[0,axis] != padded_box[0,axis]:
            leading_halo_box = padded_box.copy()
            leading_halo_box[1, axis] = orig_box[0,axis]
            halo_boxes.append(leading_halo_box)

        if orig_box[1,axis] != padded_box[1,axis]:
            trailing_halo_box = padded_box.copy()
            trailing_halo_box[0, axis] = orig_box[1,axis]
            halo_boxes.append(trailing_halo_box)

    assert halo_boxes, \
        "How could halo_boxes be empty if there was padding needed?"

    for halo_box in halo_boxes:
        # Retrieve padding data for one halo side
        halo_volume = volume_accessor_func(halo_box)
        
        # Overwrite in the final padded volume
        halo_box_within_padded = halo_box - padded_box[0]
        overwrite_subvol(padded_volume, halo_box_within_padded, halo_volume)

    return Brick( brick.logical_box, padded_box, padded_volume )


def apply_labelmap_to_bricks(bricks, labelmap_config, working_dir, unpersist_original=False):
    """
    Relabel the bricks with a labelmap.
    If the given config specifies not labelmap, then the original is returned.

    bricks: RDD of Bricks
    
    labelmap_config: config dict, adheres to LabelMapSchema
    
    working_dir: If labelmap is from a gbucket, it will be downlaoded to working_dir.
    
    unpersist_original: If True, unpersist (or replace) the input.
                     Otherwise, the caller is free to unpersist the returned
                     bricks without affecting the input.
    
    Returns:
        remapped_bricks - Already computed and persisted.
    """
    path = labelmap_config["file"]
    if not path:
        if not unpersist_original:
            # The caller wants to be sure that the result can be
            # unpersisted safely without affecting the bricks he passed in,
            # so we return a *new* RDD, even though it's just a copy of the original.
            return rt.map( lambda brick: brick, bricks )
        return bricks

    mapping_pairs = load_labelmap(labelmap_config, working_dir)
    remapped_bricks = apply_label_mapping(bricks, mapping_pairs)

    if unpersist_original:
        unpersist(bricks)

    return remapped_bricks


def apply_label_mapping(bricks, mapping_pairs):
    """
    Given an RDD of bricks (of label data) and a pre-loaded labelmap in
    mapping_pairs [[orig,new],[orig,new],...],
    apply the mapping to the bricks.
    
    bricks:
        RDD of Bricks containing label volumes
    
    mapping_pairs:
        Mapping as returned by load_labelmap.
        An ndarray of the form:
            [[orig,new],
             [orig,new],
             ... ],
    """
    from dvidutils import LabelMapper
    def remap_bricks(partition_bricks):
        domain, codomain = mapping_pairs.transpose()
        mapper = LabelMapper(domain, codomain)
        
        partition_bricks = list(partition_bricks)
        for brick in partition_bricks:
            # TODO: Apparently LabelMapper can't handle non-contiguous arrays right now.
            #       (It yields incorrect results)
            #       Check to see if this is still a problem in the latest version of xtensor-python.
            brick.volume = np.asarray( brick.volume, order='C' )
            
            mapper.apply_inplace(brick.volume, allow_unmapped=True)
        return partition_bricks
    
    # Use mapPartitions (instead of map) so LabelMapper can be constructed just once per partition
    remapped_bricks = rt.map_partitions( remap_bricks, bricks )
    persist_and_execute(remapped_bricks, f"Remapping bricks", logger)
    return remapped_bricks


def realign_bricks_to_new_grid(new_grid, original_bricks):
    """
    Given a list/RDD of Bricks which are tiled over some original grid,
    chop them up and reassemble them into a new list/RDD of Bricks,
    tiled according to the given new_grid.
    
    Requires data shuffling.
    
    Returns: RDD (or iterable):
        [ (logical_box, Brick),
          (logical_box, Brick), ...]
    """
    # For each original brick, split it up according
    # to the new logical box destinations it will map to.
    new_logical_boxes_and_brick_fragments = rt.flat_map( partial(split_brick, new_grid), original_bricks )

    # Group fragments according to their new homes
    grouped_brick_fragments = rt.group_by_key( new_logical_boxes_and_brick_fragments )
    
    # Re-assemble fragments into the new grid structure.
    new_logical_boxes_and_bricks = rt.map_values(assemble_brick_fragments, grouped_brick_fragments)
    new_logical_boxes_and_bricks = rt.filter( lambda key_brick: key_brick[1] is not None, new_logical_boxes_and_bricks )
    return new_logical_boxes_and_bricks


def split_brick(new_grid, original_brick):
    """
    Given a single brick and a new grid to which its data should be redistributed,
    split the brick into pieces, indexed by their NEW grid locations.
    
    The brick fragments are returned as Bricks themselves, but with relatively
    small volume and physical_box members.
    
    Note: It is probably a mistake to call this function for Bricks which have
          a larger physical_box than logical_box, so that is currently forbidden.
          (It would work here, but it implies that you will end up with some voxels
          represented multiple times in a given RDD of Bricks, with undefined results
          as to which ones are kept after you consolidate them into a new alignment.
          
          However, the reverse is permitted, i.e. it is permitted for the DESTINATION
          grid to use a halo, in which case some pixels in the original brick will be
          duplicated to multiple destinations.
    
    Returns: [(box,Brick), (box, Brick), ....],
            where each Brick is a fragment (to be assembled later into the new grid's bricks),
            and 'box' is the logical_box of the Brick into which this fragment should be assembled.
    """
    new_logical_boxes_and_fragments = []
    
    # Forbid out-of-bounds physical_boxes. (See note above.)
    assert ((original_brick.physical_box[0] >= original_brick.logical_box[0]).all() and
            (original_brick.physical_box[1] <= original_brick.logical_box[1]).all())
    
    # Iterate over the new boxes that intersect with the original brick
    for destination_box in boxes_from_grid(original_brick.physical_box, new_grid, include_halos=True):
        # Physical intersection of original with new
        split_box = box_intersection(destination_box, original_brick.physical_box)
        
        # Extract portion of original volume data that belongs to this new box
        split_box_internal = split_box - original_brick.physical_box[0]
        fragment_vol = extract_subvol(original_brick.volume, split_box_internal)

        # Subtract out halo to get logical_box
        new_logical_box = destination_box - (-new_grid.halo_shape, new_grid.halo_shape)

        fragment_brick = Brick(new_logical_box, split_box, fragment_vol)
        fragment_brick.compress()

        # Append key (the new_logical_box, but with a special type and hash,
        # to avoid bad collisions with the default spark hash function),
        # and new brick fragment, to be assembled into the final brick in a later stage.
        key = rt.tuple_with_hash( box_as_tuple(new_logical_box) )
        key.set_hash( hash(tuple(new_logical_box[0] / new_grid.block_shape)) )
        new_logical_boxes_and_fragments.append( (key, fragment_brick) )

    return new_logical_boxes_and_fragments


def assemble_brick_fragments( fragments ):
    """
    Given a list of Bricks with identical logical_boxes, splice their volumes
    together into a final Brick that contains a full volume containing all of
    the fragments.
    
    Note: Brick 'fragments' are also just Bricks, whose physical_box does
          not cover the entire logical_box for the brick.
    
    Each fragment's physical_box indicates where that fragment's data
    should be located within the final returned Brick.
    
    Returns: A Brick containing the data from all fragments,
            UNLESS the fully assembled fragments would not intersect
            with the Brick's own logical_box (i.e. all fragments fall
            within the halo), in which case None is returned.
    
    Note: If the fragment physical_boxes are not disjoint, the results
          are undefined.
    """
    fragments = list(fragments)

    # All logical boxes must be the same
    logical_boxes = np.asarray([frag.logical_box for frag in fragments])
    assert (logical_boxes == logical_boxes[0]).all(), \
        "Cannot assemble brick fragments from different logical boxes. "\
        "They belong to different bricks!"
    final_logical_box = logical_boxes[0]

    # The final physical box is the min/max of all fragment physical extents.
    physical_boxes = np.array([frag.physical_box for frag in fragments])
    assert physical_boxes.ndim == 3 # (N, 2, Dim)
    assert physical_boxes.shape == ( len(fragments), 2, final_logical_box.shape[1] )
    
    final_physical_box = np.asarray( ( np.min( physical_boxes[:,0,:], axis=0 ),
                                       np.max( physical_boxes[:,1,:], axis=0 ) ) )

    interior_box = box_intersection(final_physical_box, final_logical_box)
    if (interior_box[1] - interior_box[0] < 1).any():
        # All fragments lie completely within the halo
        return None

    final_volume_shape = final_physical_box[1] - final_physical_box[0]
    dtype = fragments[0].volume.dtype

    final_volume = np.zeros(final_volume_shape, dtype)

    for frag in fragments:
        internal_box = frag.physical_box - final_physical_box[0]
        overwrite_subvol(final_volume, internal_box, frag.volume)

        # Destroy original to save RAM
        frag.destroy()

    brick = Brick( final_logical_box, final_physical_box, final_volume )
    brick.compress()
    return brick




