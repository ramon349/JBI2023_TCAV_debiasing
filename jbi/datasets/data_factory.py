
class DatasetRegister: 
    __data = {}
    @classmethod
    def register(cls,cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name]=cls_obj
            return cls_obj
        return decorator
    @classmethod
    def get_dataset(cls,key):
        return cls.__data[key]
    @classmethod
    def num_datasets(cls):
        return len(cls.__data)
    @classmethod
    def get_datasets(cls):
        return cls.__data.keys()

def get_dataset(conf):
    return DatasetRegister.get_dataset(conf['dataset'])