def test_attack(args, model, test_loader, adv_perturb):
    total = correct = adv_correct = 0
    model.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = model(images)
        correct += float((pred.argmax(dim=1) == labels).sum())

        adv_images = adv_perturb(args, images, model, labels)
        adv_pred = model(adv_images)
        adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
        total += len(labels)
    return correct, adv_correct, total
