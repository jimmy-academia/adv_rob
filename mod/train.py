from utils import check

def prepare_tokenembedder(args, tokenembedder, train_loader, test_loader):
    if args.precluster_method == 'kmeans':
        for images, __ in train_loader:
            patches = tokenembedder.split_patch(images)
            check()
    pass

def perpare_classifier(args, tokenembedder, classifier, train_loader, test_loader):
    pass