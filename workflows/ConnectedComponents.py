"""Framework for large-scale connected components over an ROI."""
import textwrap
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
import DVIDSparkServices
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class ConnectedComponents(DVIDWorkflow):
    # schema for creating segmentation
    Schema = textwrap.dedent("""\
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create connected components from segmentation",
      "type": "object",
      "properties": {
        "dvid-info" : {
          "type": "object",
          "properties": {
            "dvid-server": {
              "description": "location of DVID server",
              "type": "string",
              "minLength": 1,
              "property": "dvid-server"
            },
            "uuid": {
              "description": "version node to store segmentation",
              "type": "string",
              "minLength": 1
            },
            "roi": {
              "description": "region of interest to segment",
              "type": "string",
              "minLength": 1
            },
            "segmentation": {
              "description": "location for segmentation result",
              "type": "string",
              "minLength": 1
            },
            "newsegmentation": {
              "description": "location for segmentation result",
              "type": "string",
              "minLength": 1
            }
          },
          "required": ["dvid-server", "uuid", "roi", "segmentation", "newsegmentation"],
          "additionalProperties": false
        },
        "options" : {
          "type": "object",
          "properties": {
            "chunk-size": {
              "description": "Size of blocks to process independently (and then stitched together).",
              "type": "integer",
              "default": 512
            },
            "debug": {
              "description": "Enable certain debugging functionality.  Mandatory for integration tests.",
              "type": "boolean",
              "default": false
            }
          },
          "additionalProperties": false
        }
      }
    }
    """)

    # assume blocks are 32x32x32
    blocksize = 32

    # overlap between chunks
    overlap = 2

    def __init__(self, config_filename):
        # ?! set number of cpus per task to 2 (make dynamic?)
        super(ConnectedComponents, self).__init__(config_filename, self.Schema, "ConnectedComponents")


    # (stitch): => flatmap to boundary, id, cropped labels => reduce to common boundaries maps
    # => flatmap substack and boundary mappings => take ROI max id collect, join offsets with boundary
    # => join offsets and boundary mappings to persisted ROI+label, unpersist => map labels
    # (write): => for each row
    def execute(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel
        from DVIDSparkServices.reconutils.Segmentor import Segmentor
        from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
        import numpy

        self.chunksize = self.config_data["options"]["chunk-size"]

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi(
                self.config_data["dvid-info"]["roi"],
                self.chunksize, self.overlap/2, True)
        distsubvolumes.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # grab seg chunks 
        seg_chunks = self.sparkdvid_context.map_labels64(distsubvolumes,
                self.config_data["dvid-info"]["segmentation"],
                self.overlap/2, self.config_data["dvid-info"]["roi"], True)
        # pass substack with labels (no shuffling)
        seg_chunks2 = distsubvolumes.join(seg_chunks) 
        distsubvolumes.unpersist()
        
        # run connected components
        def connected_components(seg_chunk):
            substack, seg_c = seg_chunk
            seg = seg_c.deserialize()
            from DVIDSparkServices.reconutils.morpho import split_disconnected_bodies
            seg2, dummy = split_disconnected_bodies(seg)

            # renumber from one 
            vals = numpy.unique(seg2)
            remap = {}
            index = 1
            remap[0] = 0
            for val in vals:
                if val != 0:
                    remap[val] = index
                    index += 1
            vectorized_relabel = numpy.frompyfunc(remap.__getitem__, 1, 1)
            seg2 = vectorized_relabel(seg2).astype(numpy.uint32)

           
            substack.set_max_id( seg2.max() )
            return (substack, CompressedNumpyArray(seg2))

        seg_chunks_cc = seg_chunks2.mapValues(connected_components)

        from DVIDSparkServices.reconutils.Segmentor import Segmentor
        # stitch the segmentation chunks
        # (preserves initial partitioning)
        from DVIDSparkServices.reconutils.morpho import stitch 
        mapped_seg_chunks = stitch(self.sparkdvid_context.sc, seg_chunks_cc)
       

        # coalesce to fewer partitions (!!TEMPORARY SINCE THERE ARE WRITE BANDWIDTH LIMITS TO DVID)
        #mapped_seg_chunks = mapped_seg_chunks.coalesce(125)

        # write data to DVID
        self.sparkdvid_context.foreach_write_labels3d(self.config_data["dvid-info"]["newsegmentation"], mapped_seg_chunks)
        self.logger.write_data("Wrote DVID labels") # write to logger after spark job


    @staticmethod
    def dumpschema():
        return ConnectedComponents.Schema
