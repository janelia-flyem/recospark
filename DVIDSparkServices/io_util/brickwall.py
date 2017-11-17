import numpy as np

from DVIDSparkServices import rddtools as rt

from .brick import Grid, generate_bricks_from_volume_source, realign_bricks_to_new_grid, pad_brick_data_from_volume_source

class BrickWall:
    """
    Manages a (lazy) set of bricks within a Grid.
    Mostly just a convenience wrapper to simplify pipelines of transformations over RDDs of bricks.
    """
    
    ##
    ## Operations
    ##

    def realign_to_new_grid(self, new_grid):
        """
        Chop upand the Bricks in this BrickWall reassemble them into a new BrickWall,
        tiled according to the given new_grid.
        
        Note: Requires data shuffling.
        
        Returns: A a new BrickWall, with a new internal RDD for bricks.
        """
        new_logical_boxes_and_bricks = realign_bricks_to_new_grid( new_grid, self.bricks )
        new_wall = BrickWall( self.bounding_box, self.grid, _bricks=new_logical_boxes_and_bricks.values() )
        return new_wall

    def fill_missing(self, volume_accessor_func, padding_grid=None):
        """
        For each brick whose physical_box does not extend to all edges of its logical_box,
        fill the missing space with data from the given volume accessor.
        
        Args:
            volume_accessor_func:
                See __init__, above.
            
            padding_grid:
                (Optional.) Need not be identical to the BrickWall's native grid,
                but must divide evenly into it. If not provided, the native grid is used.
        """
        if padding_grid is None:
            padding_grid = self.grid
            
        def pad_brick(brick):
            return pad_brick_data_from_volume_source(padding_grid, volume_accessor_func, brick)
        
        padded_bricks = rt.map( pad_brick, self.bricks )
        return padded_bricks

    ##
    ## Convenience Constructor
    ##

    @classmethod
    def from_volume_service(cls, volume_service, sc=None, target_partition_size_voxels=None):
        grid = Grid(volume_service.preferred_message_shape, (0,0,0))
        return BrickWall( volume_service.bounding_box_zyx,
                          grid,
                          volume_service.get_subvolume,
                          sc,
                          target_partition_size_voxels )

    ##
    ## Generic Constructor
    ##

    def __init__(self, bounding_box, grid, volume_accessor_func=None, sc=None, target_partition_size_voxels=None, _bricks=None):
        """
        Generic constructor, taking an arbitrary volume_accessor_func.
        Specific convenience constructors for various DVID/Brainmaps/slices sources are below.
        
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
        """
        self.grid = grid
        self.bounding_box = bounding_box

        if _bricks:
            assert sc is None
            assert target_partition_size_voxels is None
            assert volume_accessor_func is None
            self.bricks = _bricks
        else:
            assert volume_accessor_func is not None
            rdd_partition_length = None
            if target_partition_size_voxels:
                block_size_voxels = np.prod(grid.block_shape)
                rdd_partition_length = target_partition_size_voxels // block_size_voxels
            self.bricks = generate_bricks_from_volume_source(bounding_box, grid, volume_accessor_func, sc, rdd_partition_length)
