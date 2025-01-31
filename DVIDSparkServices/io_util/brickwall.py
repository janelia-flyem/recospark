import numpy as np
import scipy.ndimage

from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.util import num_worker_nodes, cpus_per_worker
from dvidutils import downsample_labels

from neuclease.util import Grid, SparseBlockMask
from .brick import ( Brick, generate_bricks_from_volume_source,
                     realign_bricks_to_new_grid, pad_brick_data_from_volume_source, apply_labelmap_to_bricks )

class BrickWall:
    """
    Manages a (lazy) set of bricks within a Grid.
    Mostly just a convenience wrapper to simplify pipelines of transformations over RDDs of bricks.
    """
    ##
    ## Generic Constructor (See also: convenience constructors below) 
    ##

    def __init__(self, bounding_box, grid, bricks):
        """
        bounding_box: (start, stop)
        grid: Grid
        bricks: RDD of Bricks
        """
        self.bounding_box = np.array(bounding_box)
        self.grid = grid
        self.bricks = bricks
    
    ##
    ## Convenience Constructors
    ##

    @classmethod
    def from_accessor_func(cls, bounding_box, grid, volume_accessor_func=None, sc=None, target_partition_size_voxels=None, sparse_boxes=None, lazy=False):
        """
        Convenience constructor, taking an arbitrary volume_accessor_func.
        
        Args:
            bounding_box:
                (start, stop)
     
            grid:
                Grid (see brick.py)
     
            volume_accessor_func:
                Callable with signature: f(box) -> ndarray
                Note: The callable will be unpickled only once per partition, so initialization
                      costs after unpickling are only incurred once per partition.
     
            sc:
                SparkContext. If provided, an RDD is returned.  Otherwise, returns an ordinary Python iterable.
     
            target_partition_size_voxels:
                Optional. If provided, the RDD partition lengths (i.e. the number of bricks per RDD partition)
                will be chosen to have (approximately) this many total voxels in each partition.
            
            sparse_boxes:
                A list of (physical) sparse boxes indicating which bricks should actually be present in the BrickWall.
                If not provided, all bricks within the bounding_box will be present. 

            lazy:
                If True, the bricks' data will not be created until their 'volume' member is first accessed.
        """
        if target_partition_size_voxels is None:
            if sc:
                num_threads = num_worker_nodes() * cpus_per_worker()
            else:
                # See RDDtools -- for now, non-spark pseudo-RDDs are just a single partition.
                num_threads = 1

            if sparse_boxes is None:
                total_voxels = np.prod(bounding_box[1] - bounding_box[0])
            else:
                if not hasattr(sparse_boxes, '__len__'):
                    sparse_boxes = list(sparse_boxes)
                total_voxels = sum( map(lambda physbox: np.prod(physbox[1] - physbox[0]), sparse_boxes ) )
            
            voxels_per_thread = total_voxels / num_threads
            target_partition_size_voxels = (voxels_per_thread // 2) # Arbitrarily aim for 2 partitions per thread

        block_size_voxels = np.prod(grid.block_shape)
        rdd_partition_length = target_partition_size_voxels // block_size_voxels

        bricks = generate_bricks_from_volume_source(bounding_box, grid, volume_accessor_func, sc, rdd_partition_length, sparse_boxes, lazy)
        return BrickWall( bounding_box, grid, bricks )


    @classmethod
    def from_volume_service(cls, volume_service, scale=0, bounding_box_zyx=None, sc=None, target_partition_size_voxels=None, sparse_block_mask=None, lazy=False):
        """
        Convenience constructor, initialized from a VolumeService object.
        
        Args:
            volume_service:
                An instance of a VolumeService
        
            bounding_box_zyx:
                (start, stop) Optional.
                Bounding box to restrict the region of fetched blocks, always
                specified in FULL-RES coordinates, even if you are passing scale > 0
                If not provided, volume_service.bounding_box_zyx is used.
     
            scale:
                Brick data will be fetched at this scale.
                (Note: The bricks' sizes will still be the the full volume_service.preferred_message_shape,
                       but the overall bounding-box of the BrickWall be scaled down.) 
     
            sc:
                SparkContext. If provided, an RDD is returned.  Otherwise, returns an ordinary Python iterable.
     
            target_partition_size_voxels:
                Optional. If provided, the RDD partition lengths (i.e. the number of bricks per RDD partition)
                will be chosen to have (approximately) this many total voxels in each partition.
            
            sparse_block_mask:
                Instance of SparseBlockMask
            
            lazy:
                If True, the bricks' data will not be created until their 'volume' member is first accessed.
        """
        grid = Grid(volume_service.preferred_message_shape, (0,0,0))
        
        if bounding_box_zyx is None:
            bounding_box_zyx = volume_service.bounding_box_zyx
        
        if scale == 0:
            downsampled_box = bounding_box_zyx
        else:
            full_box = bounding_box_zyx
            downsampled_box = np.zeros((2,3), dtype=int)
            downsampled_box[0] = full_box[0] // 2**scale # round down
            
            # Proper downsampled bounding-box would round up here...
            #downsampled_box[1] = (full_box[1] + 2**scale - 1) // 2**scale
            
            # ...but some some services probably don't do that, so we'll
            # round down to avoid out-of-bounds errors for higher scales. 
            downsampled_box[1] = full_box[1] // 2**scale

        sparse_boxes = None
        if sparse_block_mask is not None:
            assert isinstance(sparse_block_mask, SparseBlockMask)
            assert scale == 0, "FIXME: I don't think the sparse feature works with scales other than 0."
            sparse_boxes = sparse_block_mask.sparse_boxes(grid)

        return BrickWall.from_accessor_func( downsampled_box,
                                             grid,
                                             lambda box: volume_service.get_subvolume(box, scale),
                                             sc,
                                             target_partition_size_voxels,
                                             sparse_boxes,
                                             lazy )


    ##
    ## Operations
    ##
    def drop_empty(self):
        """
        Remove all empty (completely zero) bricks from the BrickWall.
        """
        filtered_bricks = rt.filter(lambda brick: brick.volume.any(), self.bricks)
        return BrickWall( self.bounding_box, self.grid, filtered_bricks )


    def realign_to_new_grid(self, new_grid):
        """
        Chop upand the Bricks in this BrickWall reassemble them into a new BrickWall,
        tiled according to the given new_grid.
        
        Note: Requires data shuffling.
        
        Returns: A a new BrickWall, with a new internal RDD for bricks.
        """
        new_logical_boxes_and_bricks = realign_bricks_to_new_grid( new_grid, self.bricks )
        new_wall = BrickWall( self.bounding_box, new_grid, rt.values(new_logical_boxes_and_bricks) )
        return new_wall


    def fill_missing(self, volume_accessor_func, padding_grid=None):
        """
        For each brick whose physical_box does not extend to all edges of its logical_box,
        fill the missing space with data from the given volume accessor.
        
        Args:
            volume_accessor_func:
                Callable with signature: f(box) -> ndarray
                Note: The callable will be unpickled only once per partition, so initialization
                      costs after unpickling are only incurred once per partition.
            
            padding_grid:
                (Optional.) Need not be identical to the BrickWall's native grid,
                but must divide evenly into it. If not provided, the native grid is used.
        """
        if padding_grid is None:
            padding_grid = self.grid
            
        def pad_brick(brick):
            return pad_brick_data_from_volume_source(padding_grid, volume_accessor_func, brick)
        
        padded_bricks = rt.map( pad_brick, self.bricks )
        new_wall = BrickWall( self.bounding_box, self.grid, padded_bricks )
        return new_wall


    def apply_labelmap(self, labelmap_config, working_dir, unpersist_original=False):
        """
        Relabel the bricks in this BrickWall with a labelmap.
        If the given config specifies not labelmap, then the original is returned.
        
        NOTE: It is rarely necessary to call this function, since labelmaps can be
              specified in a VolumeService config, in which case the labelmap is
              automatically applied during reading/writing.
    
        bricks: RDD of Bricks
        
        labelmap_config: config dict, adheres to LabelMapSchema
        
        working_dir: If labelmap is from a gbucket, it will be downlaoded to working_dir.
        
        unpersist_original: If True, unpersist (or replace) the input.
                            Otherwise, the caller is free to unpersist the returned
                            BrickWall without affecting the input.
        
        Returns:
            A new BrickWall, with remapped bricks.
        """
        remapped_bricks = apply_labelmap_to_bricks(self.bricks, labelmap_config, working_dir, unpersist_original)
        return BrickWall( self.bounding_box, self.grid, remapped_bricks )


    def translate(self, offset_zyx):
        """
        Translate all bricks by the given offset.
        Does not change the brick data, just the logical/physical boxes.
        
        Also, translates the bounding box and grid.
        """
        def translate_brick(brick):
            return Brick( brick.logical_box + offset_zyx,
                          brick.physical_box + offset_zyx,
                          brick.volume )

        translated_bricks = rt.map( translate_brick, self.bricks )
        
        new_bounding_box = self.bounding_box + offset_zyx
        new_grid = Grid( self.grid.block_shape, self.grid.offset + offset_zyx )
        return BrickWall( new_bounding_box, new_grid, translated_bricks )

    
    def persist_and_execute(self, description, logger=None, storage=None):
        self.bricks = rt.persist_and_execute( self.bricks, description, logger, storage )
    
    
    def unpersist(self):
        rt.unpersist(self.bricks)


    def downsample(self, block_shape, method):
        assert block_shape[0] == block_shape[1] == block_shape[2], \
            "Currently, downsampling must be isotropic"
        assert method in ['label', 'grayscale']

        factor = block_shape[0]
        def downsample_brick(brick):
            assert (brick.physical_box % factor == 0).all()
            assert (brick.logical_box % factor == 0).all()
        
            if method == 'grayscale':
                downsampled_volume = scipy.ndimage.interpolation.zoom(brick.volume, 1/factor, mode='reflect')
            elif method == 'label':
                # Old: Python downsampling
                # downsample_3Dlabels(brick.volume)
            
                # Newer: Numba downsampling
                #downsampled_volume, _ = downsample_labels_3d_suppress_zero(brick.volume, (2,2,2), brick.physical_box)
            
                # Even Newer: C++ downsampling (note: only works on aligned data.)
                # For consistency with DVID's on-demand downsampling, we suppress 0 pixels.
                downsampled_volume = downsample_labels(brick.volume, factor, suppress_zero=True)
            else:
                assert False
        
            downsampled_logical_box = brick.logical_box // factor
            downsampled_physical_box = brick.physical_box // factor
            
            return Brick(downsampled_logical_box, downsampled_physical_box, downsampled_volume)

        new_bounding_box = self.bounding_box // factor
        new_grid = Grid( self.grid.block_shape // factor, self.grid.offset // factor )
        new_bricks = rt.map( downsample_brick, self.bricks )
        
        return BrickWall( new_bounding_box, new_grid, new_bricks )


    def copy(self):
        """
        Return a duplicate of this BrickWall, with a new bricks RDD (which not persisted).
        """
        return BrickWall( self.bounding_box, self.grid, rt.map( lambda x:x, self.bricks ) )



