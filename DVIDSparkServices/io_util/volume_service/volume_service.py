from abc import ABCMeta, abstractmethod, abstractproperty

class VolumeService(metaclass=ABCMeta):

    SUPPORTED_SERVICES = ['dvid', 'brainmaps', 'n5', 'slice-files']

    @abstractproperty
    def dtype(self):
        raise NotImplementedError

    @abstractproperty
    def preferred_message_shape(self):
        raise NotImplementedError

    @abstractproperty
    def block_width(self):
        raise NotImplementedError

    @classmethod
    def create_from_config( cls, volume_config, config_dir, resource_manager_client=None ):
        from .dvid_volume_service import DvidVolumeService
        from .brainmaps_volume_service import BrainMapsVolumeServiceReader
        from .n5_volume_service import N5VolumeServiceReader
        from .slice_files_volume_service import SliceFilesVolumeServiceReader

        VolumeService._remove_default_service_configs(volume_config)

        service_keys = set(volume_config.keys()).intersection( set(VolumeService.SUPPORTED_SERVICES) )
        if len(service_keys) != 1:
            raise RuntimeError(f"Unsupported service (or too many specified): {service_keys}")
        
        if "dvid" in volume_config:
            return DvidVolumeService( volume_config, resource_manager_client )
        if "brainmaps" in volume_config:
            return BrainMapsVolumeServiceReader( volume_config, resource_manager_client )
        if "n5" in volume_config:
            return N5VolumeServiceReader( volume_config, config_dir )
        if "slice-files" in volume_config:
            return SliceFilesVolumeServiceReader( volume_config )
    
        assert False, "Shouldn't get here."

    @classmethod
    def _remove_default_service_configs(cls, volume_config):
        """
        The validate_and_inject_defaults() function will insert default
        settings for all possible service configs, but we are only interested
        in the one that the user actually wrote.
        Fortunately, that function places a special hint 'from_default' on the config
        dict to make it easy to figure out which configs were completely default-generated.
        """
        for key in VolumeService.SUPPORTED_SERVICES:
            if key in volume_config and hasattr(volume_config[key], 'from_default') and volume_config[key].from_default:
                del volume_config[key]

class VolumeServiceReader(VolumeService):
    
    @abstractproperty
    def bounding_box_zyx(self):
        raise NotImplementedError

    @abstractmethod
    def get_subvolume(self, box_zyx, scale=0):
        raise NotImplementedError

class VolumeServiceWriter(VolumeService):

    @abstractmethod
    def write_subvolume(self, subvolume, offset_zyx, scale):
        raise NotImplementedError
