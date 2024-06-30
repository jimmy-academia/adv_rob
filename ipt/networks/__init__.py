from ipt.networks.patch_makers import DisjointPatchMaker
from ipt.networks.tokenizers import MLPTokenizer
from ipt.networks.iptresnet import IPTResnet
from ipt.networks.decoder import Decoder
from ipt.networks.aptnet import APTNet

from ipt.networks.cnns import Dummy, SmallClassifier, AptSizeClassifier


def build_ipt_model(args):
    if args.config['model']['patcher_type'] == 'conv_disjoint':
        PatchMaker = DisjointPatchMaker(args)
    if args.config['model']['tokenizer_type'] == 'mlp':
        Tokenizer = MLPTokenizer(args.patch_numel, args.vocab_size, args.config['model']['tokenizer_hidden_layers'])
    return IPTResnet(args, PatchMaker, Tokenizer)
