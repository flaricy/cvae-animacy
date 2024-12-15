import inspect
import omegaconf
from omegaconf import OmegaConf
from .misc import is_seq_of

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope()
        
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, items={self._module_dict})"
        return format_str
        
    def get(self, key):
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            if real_key in self._module_dict:
                return self._module_dict[real_key]
            
    def build(self, cfg, use_omegaconf=True):
        return self.build_from_cfg(cfg=cfg, registry=self, use_omegaconf=use_omegaconf)
    
    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError(f"module_class must be a class, but got {type(module_class)}")
        
        if module_name is None:
            module_name = module_class.__name__
            
        if isinstance(module_name, str):
            module_name = [module_name]
            
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered in {self._name}")
            self._module_dict[name] = module_class
            
    def register_module(self, name=None, force=False, module=None):
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        
        if not name is None and not isinstance(name, str) and not is_seq_of(name, str):
            raise TypeError(f"name must be 'None' or a string or a sequence of strings, but got {type(name)}")

        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module 
        
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls
        
        return _register
            
    @staticmethod
    def split_scope_key(key):
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key
    
    @staticmethod
    def build_from_cfg(cfg, registry, use_omegaconf):
        if not isinstance(cfg, omegaconf.dictconfig.DictConfig) and not isinstance(cfg, dict):
            raise TypeError("cfg must be an OmegaConf DictConfig")
        if "type" not in cfg:
            raise KeyError("cfg must contain a 'type' key")
        
        if not isinstance(registry, Registry):
            raise TypeError("registry must be an instance of Registry")
        
        args = cfg.copy()
        
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"Unrecognized type {obj_type}")
        else:
            raise TypeError("type must be a string")
        
        try:
            if use_omegaconf:
                return obj_cls(args)
            else:
                return obj_cls(**args)
        except Exception as e:
            raise type(e)(f"{obj_cls.__name__} instantiation failed: {str(e)}")
        
    @staticmethod
    def infer_scope():
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]
    
    @property
    def name(self):
        return self._name
    
    @property
    def scope(self):
        return self._scope
    
    @property
    def module_dict(self):
        return self._module_dict


