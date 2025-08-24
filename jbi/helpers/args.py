
import argparse
import json 
from collections import deque 

from ..datasets.data_factory import DatasetRegister 
from ..trainers.trainer_factory import TrainerRegister
from ..models.model_factory import ModelRegister 

class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        data_dict = json.load(values)
        arg_list = deque()
        action_dict = {e.option_strings[0]: e for e in parser._actions}
        for i, e in enumerate(data_dict):
            arg_list.extend(self.__build_parse_arge__(e, data_dict, action_dict))
        parser.parse_args(arg_list, namespace=namespace)

    def __build_parse_arge__(self, arg_key, arg_dict, file_action):
        arg_name = f"--{arg_key}"
        arg_val = str(arg_dict[arg_key]).replace(
            "'", '"'
        )  # list of text need to be modified so they can be parsed properly
        try:
            file_action[arg_name].required = False
        except:
            raise KeyError(
                f"The Key {arg_name} is not an expected parameter. Delete it from config or update build_args method in helper_utils.configs.py"
            )
        return arg_name, arg_val
    
def get_trainer_choices():
    """ Returns the possible traienrs available when parsing config
    """
    return TrainerRegister.get_trainers()
def get_data_choices(): 
    """ Returns the possible datasets available when parsing config
    """
    return DatasetRegister.get_datasets() 

def get_model_choices():
    return ModelRegister.get_models()
def build_train_args(mode=None):
    """Parses args
    """
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model training for segmentation"
    )
    parser.add_argument(
        "--config_path", required=False, type=open, action=LoadFromFile, help="Path"
    ) 
    parser.add_argument("--csv_path",required=True,type=str)
    parser.add_argument("--transform_conf",required=True,type=json.loads)
    parser.add_argument(
        "--train_transforms",
        type=json.loads,
        required=True,
        help="List of Names of train transforms and augmentations in form [load,rotate]",
    )
    parser.add_argument(
        "--test_transforms",
        type=json.loads,
        required=True,
        help="List of Names of test transforms and augmentations in form [load] should be subset of train transforms",
    )  # TODO: asert test is subset of train excluding rands
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument('--dataset',type=str,required=True,choices=get_data_choices())
    parser.add_argument('--col_info',type=json.loads,required=True) 
    parser.add_argument("--num_workers",type=int,required=True)
    parser.add_argument("--device",type=json.loads,required=True)
    parser.add_argument("--model",type=str,required=True,choices=get_model_choices())
    parser.add_argument("--trainer_args",type=json.loads,required=True)
    parser.add_argument("--splits",type=json.loads,required=True)
    parser.add_argument("--model_parameters",type=json.loads,required=True)
    parser.add_argument("--debug",type=parse_bool,required=False)
    parser.add_argument("--model_weight",type=str,required=False)
    parser.add_argument("--weight_task",required=False,default=False,type=parse_bool)
    match mode: 
        case 'train': 
            parser.add_argument("--log_dir", type=str, required=True) 
        case 'optimize':
            parser.add_argument("--optuna_log",type=str,required=True)
            parser.add_argument('--direction',type=json.loads,required=True)
            parser.add_argument('--n_trials',type=int,required=True)
    return parser



def build_tcav_args(): 
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model training for segmentation"
    )
    parser.add_argument(
        "--config_path", required=False, type=open, action=LoadFromFile, help="Path"
    ) 
    parser.add_argument("--csv_path",required=True,type=str)
    parser.add_argument("--transform_conf",required=True,type=json.loads)
    parser.add_argument(
        "--test_transforms",
        type=json.loads,
        required=True,
        help="List of Names of test transforms and augmentations in form [load] should be subset of train transforms",
    )  # TODO: asert test is subset of train excluding rands
    parser.add_argument("--tcav_args",type=json.loads,required=True)
    parser.add_argument('--col_info',type=json.loads,required=True) 
    parser.add_argument("--model_parameters",type=json.loads,required=True)
    parser.add_argument("--model",type=str,required=True,choices=get_model_choices())
    parser.add_argument("--model_weight",type=str,required=False,default=None)
    parser.add_argument("--device",type=json.loads,required=True)
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--log_dir",type=str,required=True)
    return parser
def parse_bool(s: str): 
    if s.lower() == 'true': 
        return  True  
    if s.lower() == 'false': 
        return  False  
    else: 
        raise Exception("Typo in bool var please check")

def get_train_args():
    parser = build_train_args(mode='train')
    args = parser.parse_args()
    conf = vars(args)
    return conf

def get_optuna_params():
    parser = build_train_args(mode='optimize')
    args = parser.parse_args()
    conf = vars(args)
    return conf
def get_tcav_args():
    parser = build_tcav_args()
    args = parser.parse_args()
    conf = vars(args)
    return conf
