from typing import Dict, Type

_REGISTRY: Dict[str, type] = {}

def register_extractor(name: str):
    def _decorator(cls: Type):
        if name in _REGISTRY:
            raise KeyError(f"Extractor '{name}' already registered.")
        print(f"Registering extractor: {name} -> {cls}")
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return _decorator

def get_extractor(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Extractor '{name}' not found. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]

def list_extractors():
    return sorted(_REGISTRY.keys())
