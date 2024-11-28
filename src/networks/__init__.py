from networks.model_list import * 

def get_model(args):
    if '_' not in args.model:
        model = globals()[args.model](args)
    else:
        model = globals()['dummy_model'](args)
    return model