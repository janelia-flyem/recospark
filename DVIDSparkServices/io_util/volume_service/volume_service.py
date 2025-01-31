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

    @property
    def base_service(self):
        """
        If this service wraps another one (e.g. ScaledVolumeService, etc.),
        return the wrapped service.
        Default for 'base' services (e.g. DvidVolumeService) is to just return self.
        """
        return self

    @property
    def service_chain(self):
        """
        If this service wraps another service(s) (e.g. ScaledVolumeService, etc.),
        return the chain wrapped services, including the base service.
        If this service is a base service, self will be the only item in the list.
        """
        if hasattr(self, 'original_volume_service'):
            return [self] + self.original_volume_service.service_chain
        return [self]

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
        
        # Choose base service
        if "dvid" in volume_config:
            service = DvidVolumeService( volume_config, resource_manager_client )
        elif "brainmaps" in volume_config:
            service = BrainMapsVolumeServiceReader( volume_config, resource_manager_client )
        elif "n5" in volume_config:
            service = N5VolumeServiceReader( volume_config, config_dir )
        elif "slice-files" in volume_config:
            service = SliceFilesVolumeServiceReader( volume_config, config_dir )
        else:
            raise RuntimeError( "Unknown service type." )

        if 'labelmap' in volume_config:
            raise RuntimeError("Bad key for volume service: 'labelmap' -- did you mean 'apply-labelmap'?")

        # Wrap with labelmap service
        from . import LabelmappedVolumeService
        if ("apply-labelmap" in volume_config) and (volume_config["apply-labelmap"]["file-type"] != "__invalid__"):
            service = LabelmappedVolumeService(service, volume_config["apply-labelmap"], config_dir)

        # Wrap with transpose service
        from . import TransposedVolumeService
        if ("transpose-axes" in volume_config) and (volume_config["transpose-axes"] != TransposedVolumeService.NO_TRANSPOSE):
            service = TransposedVolumeService(service, volume_config["transpose-axes"])

        # Wrap with scaled service
        from . import ScaledVolumeService

        # Even if rescale-level == 0, we still wrap in a scaled volumeservice because
        # it enables more 'available-scales'
        if ("rescale-level" in volume_config):
            service = ScaledVolumeService(service, volume_config["rescale-level"])

        return service

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

    @abstractproperty
    def available_scales(self):
        raise NotImplementedError

    @abstractmethod
    def get_subvolume(self, box_zyx, scale=0):
        raise NotImplementedError

class VolumeServiceWriter(VolumeService):

    @abstractmethod
    def write_subvolume(self, subvolume, offset_zyx, scale):
        raise NotImplementedError

