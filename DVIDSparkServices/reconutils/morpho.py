"""Morphological operations used by recon utilities.

This module contains helper functions for various morphological
operations used throughout reconutils.

"""
import numpy
import scipy.sparse
import vigra
from DVIDSparkServices.util import select_item

try:
    from numba import jit
except ImportError:
    # Fake jit decorator if numba isn't available
    def jit(nopython=False):
        def wrapper(f):
            return f
        return wrapper

def split_disconnected_bodies(labels_orig):
    """
    Produces 3D volume split into connected components.

    This function identifies bodies that are the same label
    but are not connected.  It splits these bodies and
    produces a dict that maps these newly split bodies to
    the original body label.

    Special exception: Segments with label 0 are not relabeled.

    Args:
        labels_orig (numpy.array): 3D array of labels

    Returns:
        (labels_new, new_to_orig)

        labels_new:
            The partially relabeled array.
            Segments that were not split will keep their original IDs.
            Among split segments, the largest 'child' of a split segment retains the original ID.
            The smaller segments are assigned new labels in the range (N+1)..(N+1+S) where N is
            highest original label and S is the number of new segments after splitting.
        
        new_to_orig:
            A pseudo-minimal (but not quite minimal) mapping of labels
            (N+1)..(N+1+S) -> some subset of (1..N),
            which maps new segment IDs to the segments they came from.
            Segments that were not split at all are not mentioned in this mapping,
            for split segments, every mapping pair for the split is returned, including the k->k (identity) pair.
    """
    # Pre-allocate destination to force output dtype
    labels_consecutive = numpy.zeros_like(labels_orig, numpy.uint32)
    labels_new = numpy.empty_like(labels_orig)

    labels_consecutive, max_consecutive_label, orig_to_consecutive = \
        vigra.analysis.relabelConsecutive(labels_orig, start_label=1, out=labels_consecutive)

    max_orig = max( orig_to_consecutive.keys() )
    cons_to_orig = reverse_dict( orig_to_consecutive )
    
    labels_split = vigra.analysis.labelMultiArrayWithBackground(labels_consecutive)

    # 'orig' = original label values I..J
    # 'origWithSplits' = original label values I..J
    #                    plus new label values for the S splits (J+1)..(J+1+S)
    #
    # 'cons' = consecutive label values 1..N
    # 'consWithSplits' = consecutive label values 1..N
    #                    plus new label values for the S splits (N+1)..(N+1+S)

    split_to_consWithSplits, consWithSplits_to_cons = _split_body_mappings(labels_consecutive, labels_split)
    del labels_consecutive
    
    num_main_segments = max_consecutive_label
    num_splits = len(split_to_consWithSplits) - num_main_segments - 1 # not counting zero

    # Combine    
    consWithSplits_to_origWithSplits = reverse_dict(orig_to_consecutive)
    consWithSplits_to_origWithSplits.update(
        dict( zip( range( 1+max_consecutive_label, 1+max_consecutive_label+num_splits ),
                   range( 1+max_orig, 1+max_orig+num_splits ) ) ) )

    # split -> consWithSplits -> origWithSplits
    split_to_origWithSplits = compose_mappings( split_to_consWithSplits, consWithSplits_to_origWithSplits )

    # Remap the image: split -> origWithSplits
    vigra.analysis.applyMapping(labels_split, split_to_origWithSplits, out=labels_new)
    del labels_split

    origWithSplits_to_consWithSplits = reverse_dict( consWithSplits_to_origWithSplits )

    # origWithSplits -> consWithSplits -> cons -> orig
    origWithSplits_to_orig = compose_mappings( origWithSplits_to_consWithSplits,
                                               consWithSplits_to_cons,
                                               cons_to_orig )
    
    # Return final reverse mapping, but remove the labels that stayed the same.
    MINIMAL_origWithSplits_to_orig = dict( filter( lambda (k,v): k > max_orig, origWithSplits_to_orig.items() ) )
    
    # Update 2017-02-16:
    # Every label involved in a split must be returned in the mapping, even hasn't changed.
    split_labels = set(MINIMAL_origWithSplits_to_orig.values())
    final_mapping = dict(MINIMAL_origWithSplits_to_orig)
    for k,v in origWithSplits_to_orig.items():
        if v in split_labels:
            final_mapping[k] = v
    
    return labels_new, final_mapping


def _split_body_mappings( labels_orig, labels_split ):
    """
    Helper function for split_disconnected_bodies()
    
    Given an original label image and a connected components labeling
    of that original image that 'splits' any disconnected objects it contained,
    returns two mappings:
    
    1. A mapping 'split_to_nonconflicting' which converts labels_split into a
       volume that matches labels_orig as closely as possible:
         - Unsplit segments are mappted to their original IDs
         - For split segments:
           -- the largest segment retains the original ID
           -- the other segments are mapped to new labels,
              all of which are higher than labels_orig.max()
    
    2. A mapping 'nonconflicting_to_orig' to convert from
       split_to_consistent.values() to the set of values in labels_orig
       
    Args:
        labels_orig: A label image with CONSECUTIVE label values 1..N
        labels_split: A connected components labeling of the original image,
                      with label values 1..(N+M), assuming M splits in the original data.
                      
                      Note: Labels in this image do not need to have any other consistency
                      with labels in the original.  For example, label 1 in 'labels_orig'
                      may correspond to label 5 in 'labels_split'.
    """
    overlap_table_px = contingency_table(labels_orig, labels_split, sparse=True)
    num_orig_segments = overlap_table_px.shape[0] - 1 # (No zero label)
    num_split_segments = overlap_table_px.shape[1] - 1 # (No zero label)
    
    # For each 'orig' id, in which 'split' id did it mainly end up?
    main_split_segments = matrix_argmax(overlap_table_px, axis=1)
    
    overlap_table_px = overlap_table_px.tocsr()
    split_to_orig = dict( numpy.transpose( overlap_table_px.nonzero() )[:, ::-1] )
    split_to_orig[0] = 0

    # Convert to bool, remove the 'main' entries;
    # remaining entries are the new segments
    overlap_table_bool = overlap_table_px.astype(bool)
    for i, s in enumerate(main_split_segments):
        overlap_table_bool[i, s] = False

    # ('main' segments have the same id in the 'orig' and 'nonconflicting' label sets)
    main_split_ids_to_nonconflicting = _main_split_ids_to_orig = \
        { main_split_segments[orig] : orig for orig in range(0, 1+num_orig_segments) }

    # What are the 'non-main' IDs (i.e. new segments after the split)?
    nonmain_split_ids = numpy.unique( overlap_table_bool.nonzero()[1] )

    # Map the new split segments to new high ids, so they don't conflict with the old ones
    nonmain_split_ids_to_nonconflicting = dict( zip( nonmain_split_ids,
                                                     range( 1+num_orig_segments, 1+num_split_segments ) ) )

    # Map from split -> nonconflicting, i.e. (split -> main old/nonconflicting + split -> nonmain nonconflicting)
    split_to_nonconflicting = dict(main_split_ids_to_nonconflicting)
    split_to_nonconflicting.update( nonmain_split_ids_to_nonconflicting )
    assert len(split_to_nonconflicting) == len(split_to_orig)

    nonconflicting_to_split = reverse_dict( split_to_nonconflicting )

    # Map from nonconflicting -> split -> orig
    nonconflicting_to_orig = compose_mappings(nonconflicting_to_split, split_to_orig)
    
    assert len(split_to_nonconflicting) == len(nonconflicting_to_orig)
    return split_to_nonconflicting, nonconflicting_to_orig


def contingency_table(vol1, vol2, sparse=True):
    """
    Return a 2D array 'table' such that ``table[i,j]`` represents
    the count of overlapping pixels with value ``i`` in ``vol1``
    and value ``j`` in ``vol2``. 
    
    sparse:
        If True, return a sparse matrix (scipy.sparse.coo_matrix)
        to save RAM intead of a normal ndarray.
        (Internally, the sparse matrix entries have been deduplicated
        via sum_duplicates().)
    """
    vol1 = vol1.reshape(-1).view(numpy.int32) # Convert to int32 as a hack for efficient handling in scipy.sparse
    vol2 = vol2.reshape(-1).view(numpy.int32)
    assert vol1.shape == vol2.shape
    
    if sparse:
        ones = numpy.lib.stride_tricks.as_strided(numpy.uint32(1), vol1.shape, (0,))
        table = scipy.sparse.coo_matrix((ones, (vol1, vol2)))
        table.sum_duplicates()
        return table
    else:
        maxlabels = (vol1.max(), vol2.max())
        table = numpy.zeros( (maxlabels[0]+1, maxlabels[1]+1), dtype=numpy.uint32 )
        
        # numpy.add.at() will accumulate counts at the given array coordinates
        numpy.add.at(table, [vol1, vol2], 1 )
        return table

def matrix_argmax(m, axis=0):
    """
    Equivalent to np.argmax(table, axis=axis), but works
    for both ndarray and scipy.sparse.coo_matrix objects.
    
    Update:
        In newer versions of scipy, sparse matrix objects have
        an argmax() method, so we can delete this function once
        we upgrade our scipy dependency.
    """
    assert m.ndim == 2
    if axis == 0:
        return row_argmax(m.transpose())
    if axis == 1:
        return row_argmax(m)

def row_argmax(table):
    """
    Equivalent to np.argmax(table, axis=1), but works
    for both ndarray and scipy.sparse.coo_matrix objects.
    """
    assert isinstance(table, (numpy.ndarray, scipy.sparse.coo_matrix)), \
        "Unsupported matrix type: {}".format(type(table))
    assert table.ndim == 2
    
    if isinstance(table, numpy.ndarray):
        return numpy.argmax(table, axis=1)

    if isinstance(table, scipy.sparse.coo_matrix):
        return _sparse_row_argmax(table.col, table.row, table.data, table.shape[0])

    assert False, "Shouldn't get here..."

@jit(nopython=True)
def _sparse_row_argmax(sparse_cols, sparse_rows, sparse_data, num_dense_rows):
    """
    Helper function for row_argmax, to compute the argmax of a scipy.sparse.coo_matrix M.
    
    Args:
        sparse_cols: M.col
        sparse_rows: M.row
        sparse_data: M.data
        num_dense_rows: M.shape[0]
    
    Returns:
        Equivalent to numpy.argmax(M.toarray(), axis=1)
    """
    row_maxcols = numpy.zeros((num_dense_rows, 2), dtype=numpy.uint32)
    for i in range(sparse_cols.shape[0]):
        col = sparse_cols[i]
        row = sparse_rows[i]
        element = sparse_data[i]
        prev_max = row_maxcols[row,0]
        if element > prev_max:
            row_maxcols[row] = [element, col]
    return row_maxcols[:,1]

def reverse_dict(d):
    rev = { v:k for k,v in d.items() }
    assert len(rev) == len(d), "dict is not reversable: {}".format(d)
    return rev


def compose_mappings( *mappings ):
    """
    Given a series of mappings (dicts) that form a chain going
    from one set of labels to another, compose the chain
    together into a final dict.
    
    For example, combine mappings A->B, B->C, C->D into a new mapping A->D
    """
    AtoB = mappings[0]
    for BtoC in mappings[1:]:
        AtoC = { k: BtoC[v] for k,v in AtoB.items() }
        AtoB = AtoC
    return AtoB


def seeded_watershed(boundary, seed_threshold = 0, seed_size = 5, mask=None):
    """Extract seeds from boundary prediction and runs seeded watershed.

    Args:
        boundary (3D numpy array) = boundary predictions
        seed_threshold (int) = Add seeds where boundary prob is <= threshold
        seed_size (int) = seeds must be >= seed size
        mask (3D numpy array) = true to watershed, false to ignore
    Returns:
        3d watershed
    """
    
    from skimage import morphology as skmorph
    from numpy import bincount 

    # get seeds
    from scipy.ndimage import label as label2
    seeds = label2(boundary <= seed_threshold, output=numpy.uint32)[0]

    # remove small seeds
    if seed_size > 0:
        component_sizes = bincount(seeds.ravel())
        small_components = component_sizes < seed_size 
        small_locations = small_components[seeds]
        seeds[small_locations] = 0

    # mask out background (don't have to 0 out seeds since)
    supervoxels = skmorph.watershed(boundary, seeds,
            None, None, mask)

    return supervoxels


def stitch(sc, label_chunks):
    """
    label_chunks (RDD): [ (subvol, (seg_vol, max_id)),
                          (subvol, (seg_vol, max_id)),
                          ... ]

    Note: This function requires that label_chunks is already persist()ed in memory.
    """    
    assert label_chunks.is_cached, "You must persist() label_chunks before calling this function."
    subvolumes_rdd = select_item(label_chunks, 0)
    subvolumes = subvolumes_rdd.collect()
    max_ids = select_item(label_chunks, 1, 1).collect()

    # return all subvolumes back to the driver
    # create offset map (substack id => offset) and broadcast
    offsets = {}
    offset = numpy.uint64(0)

    for subvolume, max_id in zip(subvolumes, max_ids):
        offsets[subvolume.sv_index] = offset
        offset += max_id
    subvolume_offsets = sc.broadcast(offsets)

    # (subvol, label_vol) => [ (sv_index_1, sv_index_2), (subvol, boundary_labels)), 
    #                          (sv_index_1, sv_index_2), (subvol, boundary_labels)), ...] 
    def extract_boundaries(key_labels):
        # compute overlap -- assume first point is less than second
        def intersects(pt1, pt2, pt1_2, pt2_2):
            assert pt1 <= pt2, "point 1 greater than point 2: {} > {}".format( pt1, pt2 )
            assert pt1_2 <= pt2_2, "point 1_2 greater than point 2_2: {} > {}".format( pt1_2, pt2_2 )

            val1 = max(pt1, pt1_2)
            val2 = min(pt2, pt2_2)
            size = val2-val1
            npt1 = val1 - pt1 
            npt1_2 = val1 - pt1_2

            return npt1, npt1+size, npt1_2, npt1_2+size

        subvolume, labels = key_labels

        boundary_array = []
        
        # iterate through all ROI partners
        for partner in subvolume.local_regions:
            key1 = subvolume.sv_index
            key2 = partner[0]
            box2 = partner[1]
            if key2 < key1:
                key1, key2 = key2, key1
            
            # crop volume to overlap
            offx1, offx2, offx1_2, offx2_2 = intersects(
                            subvolume.box.x1-subvolume.border,
                            subvolume.box.x2+subvolume.border,
                            box2.x1-subvolume.border,
                            box2.x2+subvolume.border
                        )
            offy1, offy2, offy1_2, offy2_2 = intersects(
                            subvolume.box.y1-subvolume.border,
                            subvolume.box.y2+subvolume.border,
                            box2.y1-subvolume.border,
                            box2.y2+subvolume.border
                        )
            offz1, offz2, offz1_2, offz2_2 = intersects(
                            subvolume.box.z1-subvolume.border,
                            subvolume.box.z2+subvolume.border,
                            box2.z1-subvolume.border,
                            box2.z2+subvolume.border
                        )
                        
            labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2])

            # create key for boundary pair
            newkey = (key1, key2)

            # add to flat map
            boundary_array.append((newkey, (subvolume, labels_cropped)))

        return boundary_array


    # return compressed boundaries (id1-id2, boundary)
    # (subvol, labels) -> [ ( (k1, k2), (subvol, boundary_labels_1) ),
    #                       ( (k1, k2), (subvol, boundary_labels_1) ),
    #                       ( (k1, k2), (subvol, boundary_labels_1) ), ... ]
    label_vols_rdd = select_item(label_chunks, 1, 0)
    mapped_boundaries = subvolumes_rdd.zip(label_vols_rdd).flatMap(extract_boundaries) 

    # shuffle the hopefully smallish boundaries into their proper spot
    # groupby is not a big deal here since same keys will not be in the same partition
    grouped_boundaries = mapped_boundaries.groupByKey()

    # mappings to one partition (larger/second id keeps orig labels)
    # (new key, list<2>(subvolume, boundary compressed)) =>
    # (key, (subvolume, mappings))
    def stitcher(key_boundary):
        import numpy
        key, (boundary_list) = key_boundary

        # should be only two values
        if len(boundary_list) != 2:
            raise Exception("Expects exactly two subvolumes per boundary")
        # extract iterables
        boundary_list_list = []
        for item1 in boundary_list:
            boundary_list_list.append(item1)

        # order subvolume regions (they should be the same shape)
        subvolume1, boundary1 = boundary_list_list[0] 
        subvolume2, boundary2 = boundary_list_list[1] 

        if subvolume1.sv_index > subvolume2.sv_index:
            subvolume1, subvolume2 = subvolume2, subvolume1
            boundary1, boundary2 = boundary2, boundary1

        if boundary1.shape != boundary2.shape:
            raise Exception("Extracted boundaries are different shapes")
        
        # determine list of bodies in play
        z2, y2, x2 = boundary1.shape
        z1 = y1 = x1 = 0 

        # determine which interface there is touching between subvolumes 
        if subvolume1.touches(subvolume1.box.x1, subvolume1.box.x2,
                              subvolume2.box.x1, subvolume2.box.x2):
            x1 = x2/2 
            x2 = x1 + 1
        if subvolume1.touches(subvolume1.box.y1, subvolume1.box.y2,
                              subvolume2.box.y1, subvolume2.box.y2):
            y1 = y2/2 
            y2 = y1 + 1
        
        if subvolume1.touches(subvolume1.box.z1, subvolume1.box.z2,
                              subvolume2.box.z1, subvolume2.box.z2):
            z1 = z2/2 
            z2 = z1 + 1

        eligible_bodies = set(numpy.unique(boundary2[z1:z2, y1:y2, x1:x2]))
        body2body = {}

        label2_bodies = numpy.unique(boundary2)

        for body in label2_bodies:
            if body == 0:
                continue
            body2body[body] = {}

        # traverse volume to find maximum overlap
        for (z,y,x), body1 in numpy.ndenumerate(boundary1):
            body2 = boundary2[z,y,x]
            if body2 == 0 or body1 == 0:
                continue
            
            if body1 not in body2body[body2]:
                body2body[body2][body1] = 0
            body2body[body2][body1] += 1


        # create merge list 
        merge_list = []

        # merge if any overlap
        for body2, bodydict in body2body.items():
            if body2 in eligible_bodies:
                for body1, val in bodydict.items():
                    if val > 0:
                        merge_list.append([int(body1), int(body2)])
                   

        # handle offsets in mergelist
        offset1 = subvolume_offsets.value[subvolume1.sv_index] 
        offset2 = subvolume_offsets.value[subvolume2.sv_index] 
        for merger in merge_list:
            merger[0] = merger[0]+offset1
            merger[1] = merger[1]+offset2

        # return id and mappings, only relevant for stack one
        return (subvolume1.sv_index, merge_list)

    # key, mapping1; key mapping2 => key, mapping1+mapping2
    def reduce_mappings(b1, b2):
        b1.extend(b2)
        return b1

    # map from grouped boundary to substack id, mappings
    subvolume_mappings = grouped_boundaries.map(stitcher).reduceByKey(reduce_mappings)

    # reconcile all the mappings by sending them to the driver
    # (not a lot of data and compression will help but not sure if there is a better way)
    merge_list = []
    all_mappings = subvolume_mappings.collect()
    for (substack_id, mapping) in all_mappings:
        merge_list.extend(mapping)

    # make a body2body map
    body1body2 = {}
    body2body1 = {}

    for merger in merge_list:
        # body1 -> body2
        body1 = merger[0]
        if merger[0] in body1body2:
            body1 = body1body2[merger[0]]
        body2 = merger[1]
        if merger[1] in body1body2:
            body2 = body1body2[merger[1]]

        if body2 not in body2body1:
            body2body1[body2] = set()
        
        # add body1 to body2 map
        body2body1[body2].add(body1)
        # add body1 -> body2 mapping
        body1body2[body1] = body2

        if body1 in body2body1:
            for tbody in body2body1[body1]:
                body2body1[body2].add(tbody)
                body1body2[tbody] = body2

    body2body = zip(body1body2.keys(), body1body2.values())
   
    # potentially costly broadcast
    # (possible to split into substack to make more efficient but compression should help)
    master_merge_list = sc.broadcast(body2body)

    # use offset and mappings to relabel volume
    def relabel(key_label_mapping):
        import numpy

        (subvolume, labels) = key_label_mapping

        # grab broadcast offset
        offset = numpy.uint64( subvolume_offsets.value[subvolume.sv_index] )

        # check for body mask labels and protect from renumber
        fix_bodies = []
        
        labels = labels + offset 
        
        # make sure 0 is 0
        labels[labels == offset] = 0

        # create default map 
        labels_view = vigra.taggedView(labels.astype(numpy.uint64), 'zyx')
        mapping_col = numpy.sort( vigra.analysis.unique(labels_view) )
        label_mappings = dict(zip(mapping_col, mapping_col))
       
        # create maps from merge list
        for mapping in master_merge_list.value:
            if mapping[0] in label_mappings:
                label_mappings[mapping[0]] = mapping[1]

        # apply maps
        new_labels = numpy.empty_like( labels, dtype=numpy.uint64 )
        new_labels_view = vigra.taggedView(new_labels, 'zyx')
        vigra.analysis.applyMapping(labels_view, label_mappings, allow_incomplete_mapping=True, out=new_labels_view)
        return (subvolume, new_labels)

    # just map values with broadcast map
    # Potential TODO: consider fast join with partitioned map (not broadcast)
    return subvolumes_rdd.zip(label_vols_rdd).map(relabel)


