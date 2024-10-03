from networks.model_list import * 

def get_model(args):
    model = globals()[args.model](args)
    return model