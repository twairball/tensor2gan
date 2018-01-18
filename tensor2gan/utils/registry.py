from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

"""
Alot of borrowed concepts from tensor2tensor.utils.registry
"""

# Camel case to snake case 
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")

def _convert_camel_to_snake(name):
    s1 = _first_cap_re.sub(r"\1_\2", name)
    return _all_cap_re.sub(r"\1_\2", s1).lower()

def default_model_name(obj_class):
    return obj_class.__name__

def default_data_generator_name(obj_class):
    return _convert_camel_to_snake(obj_class.__name__)

# Registry
_MODELS = {}
_DATA_GENERATORS = {}

def get_models():
    return _MODELS

def get_data_generators():
    return _DATA_GENERATORS

def register_model(name=None):
    """Register a model. name defaults to class name snake-cased."""

    def decorator(model_cls, registration_name=None):
        """Registers & returns model_cls or default name.
        GANs have alot of caps so we don't use snake_case
        """
        model_name = registration_name or default_model_name(model_cls)
        if model_name in _MODELS:
            raise LookupError("Model %s already registered." % model_name)
        model_cls.REGISTERED_NAME = model_name
        _MODELS[model_name] = model_cls
        return model_cls

    # Handle if decorator was used without parens
    if callable(name):
        return decorator(name)

    return lambda model_cls: decorator(model_cls, name)

def register_data_generator(name=None):
    """Register a data generator. name defaults to class name snake-cased."""

    def decorator(data_generator_cls, registration_name=None):
        """Registers & returns data_generator_cls with registration_name or default name."""
        _name = registration_name or default_data_generator_name(data_generator_cls)
        if _name in _DATA_GENERATORS:
            raise LookupError("Data Generator %s already registered." % _name)
        data_generator_cls.REGISTERED_NAME = _name
        _DATA_GENERATORS[_name] = data_generator_cls
        return data_generator_cls

    # Handle if decorator was used without parens
    if callable(name):
        return decorator(name)

    return lambda data_generator_cls: decorator(data_generator_cls, name)
