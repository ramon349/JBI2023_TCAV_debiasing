from argparse import ArgumentParser





def _parse_args(): 
    args = ArgumentParser()
    args.add_argument("--csv_path",required=True,type=str)
    args.add_argument("--config_path",required=True,type=str)
    args.add_argument("--optuna_log_dir",required=True,type=str)
    return vars(args.parse_args())

def main(): 
    conf = _parse_args()




if __name__=='__main__':
    main() 