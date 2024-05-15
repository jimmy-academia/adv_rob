import torch
from arguments import set_arguments
from utils import set_seeds
from data import get_dataset, get_dataloader
from networks import CompactTokenEmbedder, get_resnet_model
from train import prepare_tokenembedder, perpare_classifier

def main():
    args = set_arguments()
    set_seeds(args.seed)
    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

    image_shape = (args.input_channels, args.image_size, args.image_size)
    tokenembedder = CompactTokenEmbedder(image_shape, args.patch_size, args.vocabulary_size).to(args.device)
    classifier = get_resnet_model(args.num_classes).to(args.device)

    prepare_tokenembedder(args, tokenembedder, train_loader, test_loader)
    perpare_classifier(args, tokenembedder, classifier, train_loader, test_loader)
    

if __name__ == '__main__':
    main()