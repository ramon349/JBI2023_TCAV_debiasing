from data_factories.skin_factory import get_skin_loaders
from data_factories.mammogram_factory import get_mammo_loaders
def get_factory(factory_name):
    if factory_name=='skin':
        return get_skin_loaders
    if factory_name=='mammo':
        return get_mammo_loaders
