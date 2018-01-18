from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2gan.utils.registry import get_data_generators, get_models

from tensor2gan.data_generators import cifar10, pokemon
from tensor2gan.models import dcgan, sn_dcgan

def build_model(name, config=None):
    _MODELS = get_models()
    if name not in _MODELS:
        raise LookupError("Model %s never registered. Available: %s" % (name, list(_MODELS)))
    model_cls = _MODELS[name]
    return model_cls(config)

def build_data_generator(name):
    _DATA_GENERATORS = get_data_generators()
    print("[builders] data_generators: %s" % _DATA_GENERATORS)
    if name not in _DATA_GENERATORS:
        raise LookupError("Data Generator %s never registered. Available: %s" % (name, list(_DATA_GENERATORS)))
    return _DATA_GENERATORS[name]()
