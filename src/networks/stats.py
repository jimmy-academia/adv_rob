from utils import params_to_memory
from types import SimpleNamespace
from networks import get_model

args = SimpleNamespace(dataset='cifar10', num_classes=10, channels=3, patch_size=2, image_size=32, vocab_size=10)
for modelname in ['mobilenet', 'mobileapt', 'tttbasic', 'resnetcifar', 'resnetcifarapt']:
    args.model = modelname
    model = get_model(args)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Num, whatB = params_to_memory(num_parameters)
    param_msg = f'{modelname} model param count: {num_parameters} ≈ {Num}{whatB}'
    print(param_msg)
