import copy
import logging

import numpy as np

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.io_util.brick import Grid, clipped_boxes_from_grid
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.util import num_worker_nodes, cpus_per_worker, replace_default_entries
from DVIDSparkServices.workflow.workflow import Workflow

from DVIDSparkServices.io_util.volume_service import VolumeService, GrayscaleVolumeSchema, SliceFilesVolumeSchema, SliceFilesVolumeServiceWriter

logger = logging.getLogger(__name__)

class ExportSlices(Workflow):
    OptionsSchema = copy.deepcopy(Workflow.OptionsSchema)
    OptionsSchema["additionalProperties"] = False
    OptionsSchema["properties"].update(
    {
        "slices-per-slab": {
            "description": "The volume is processed iteratively, in 'slabs' consisting of many contiguous Z-slices.\n"
                           "This setting determines the thickness of each slab. -1 means choose automatically from number of worker threads.\n"
                           "(Each worker thread processes a single Z-slice at a time.)",
            "type": "integer",
            "default": -1
        }
    })

    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Service to load raw and label data into DVID",
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "output"],
        "properties": {
            "input": GrayscaleVolumeSchema,
            "output": copy.deepcopy(SliceFilesVolumeSchema),
            "options" : OptionsSchema
        }
    }
    
    # Adjust defaults for this workflow in particular
    Schema["properties"]["output"]\
            ["properties"]["geometry"]\
              ["properties"]["message-block-shape"]["default"] = [-1, -1, 1]

    Schema["properties"]["output"]\
            ["properties"]["geometry"]\
              ["properties"]["bounding-box"]["default"] = [[-1, -1, -1], [-1, -1, -1]]


    @classmethod
    def schema(cls):
        return ExportSlices.Schema

    # name of application for DVID queries
    APPNAME = "ExportSlices".lower()

    def __init__(self, config_filename):
        super().__init__( config_filename, ExportSlices.schema(), "Export Slices" )

    def _sanitize_config(self):
        """
        Tidy up some config values, and fill in 'auto' values where needed.
        """
        input_config = self.config_data["input"]
        output_config = self.config_data["output"]

        # Initialize dummy input/output services, just to overwrite 'auto' config values as needed.
        VolumeService.create_from_config( input_config, self.config_dir )
        replace_default_entries(output_config["geometry"]["bounding-box"], input_config["geometry"]["bounding-box"])
        SliceFilesVolumeServiceWriter( output_config, self.config_dir )

        input_geometry = input_config["geometry"]
        output_geometry = output_config["geometry"]
        options = self.config_data["options"]

        # Output bounding-box must match exactly (or left as auto)
        input_bb_zyx = np.array(input_geometry["bounding-box"])[:,::-1]
        output_bb_zyx = np.array(output_geometry["bounding-box"])[:,::-1]
        assert ((output_bb_zyx == input_bb_zyx) | (output_bb_zyx == -1)).all(), \
            "Output bounding box must match the input bounding box exactly. (No translation permitted)."

        # Auto-set the output bounding-box
        output_geometry["bounding-box"] = copy.deepcopy(input_geometry["bounding-box"])
        
        assert output_config["slice-files"]["slice-xy-offset"] == [0,0], "Nonzero xy offset is meaningless for outputs."

        if options["slices-per-slab"] == -1:
            # Auto-choose a depth that keeps all threads busy with at least one slice
            brick_shape_zyx = input_geometry["message-block-shape"][::-1]
            brick_depth = brick_shape_zyx[0]
            assert brick_depth != -1
            num_threads = num_worker_nodes() * cpus_per_worker()
            threads_per_brick_layer = ((num_threads + brick_depth-1) // brick_depth) # round up
            options["slices-per-slab"] = brick_depth * threads_per_brick_layer


    def execute(self):
        self._sanitize_config()

        input_config = self.config_data["input"]
        output_config = self.config_data["output"]
        options = self.config_data["options"]

        logger.info(f"Output bounding box: {output_config['geometry']['bounding-box']}")

        mgr_client = ResourceManagerClient( options["resource-server"],
                                            options["resource-port"] )

        slice_writer = SliceFilesVolumeServiceWriter(output_config, self.config_dir)

        # Data is processed in Z-slabs
        slab_depth = options["slices-per-slab"]

        input_bb_zyx = np.array(input_config["geometry"]["bounding-box"])[:,::-1]
        _, slice_start_y, slice_start_x = input_bb_zyx[0]

        slab_shape_zyx = input_bb_zyx[1] - input_bb_zyx[0]
        slab_shape_zyx[0] = slab_depth

        slice_shape_zyx = slab_shape_zyx.copy()
        slice_shape_zyx[0] = 1

        # This grid outlines the slabs -- each grid box is a full slab
        slab_grid = Grid(slab_shape_zyx, (0, slice_start_y, slice_start_x))
        slab_boxes = list(clipped_boxes_from_grid(input_bb_zyx, slab_grid))

        for slab_index, slab_box_zyx in enumerate(slab_boxes):
            # Contruct BrickWall from input bricks
            slab_config = copy.deepcopy(input_config)
            slab_box_xyz = slab_box_zyx[:, ::-1].tolist()
            slab_config["geometry"]["bounding-box"] = slab_box_xyz
            volume_service = VolumeService.create_from_config( slab_config, self.config_dir, mgr_client )
            bricked_slab_wall = BrickWall.from_volume_service(volume_service, self.sc)

            # Force download
            bricked_slab_wall.persist_and_execute(f"Downloading slab {slab_index}/{len(slab_boxes)}: {slab_box_xyz}", logger)
            
            # Remap to slice-sized "bricks"
            sliced_grid = Grid(slice_shape_zyx, offset=slab_box_zyx[0])
            sliced_slab_wall = bricked_slab_wall.realign_to_new_grid( sliced_grid )
            sliced_slab_wall.persist_and_execute(f"Assembling slab {slab_index}/{len(slab_boxes)} slices", logger)

            # Discard original bricks
            bricked_slab_wall.unpersist()
            del bricked_slab_wall

            def write_slice(brick):
                assert (brick.physical_box == brick.logical_box).all()
                slice_writer.write_subvolume(brick.volume, brick.physical_box[0])

            # Export to PNG or TIFF, etc. (automatic via slice path extension)
            logger.info(f"Exporting slab {slab_index}/{len(slab_boxes)}", extra={"status": f"Exporting {slab_index}/{len(slab_boxes)}"})
            rt.foreach( write_slice, sliced_slab_wall.bricks )
            
            # Discard slice data
            sliced_slab_wall.unpersist()
            del sliced_slab_wall

        logger.info(f"DONE exporting {len(slab_boxes)} slabs.", extra={'status': "DONE"})
