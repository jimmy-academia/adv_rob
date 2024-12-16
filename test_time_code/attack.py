

    # sample_imgs = []
    # sample_adv_imgs = []

    # for batch_idx, (images, labels) in enumerate(tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)):
    #     images = images.to(args.device); labels = labels.to(args.device)
    #     pred = model(images)
    #     test_correct += float((pred.argmax(dim=1) == labels).sum())

    #     if args.test_time == 'none':
    #         model_copy = model
    #         tt_correct = test_correct
    #     else:
    #         if args.test_time == 'standard':
    #             raise NotImplementedError("implement test time training")
    #             model_copy = test_time_training(images)
    #         elif args.test_time == 'online':
    #             raise NotImplementedError("implement online test time training")
    #         tt_pred = model_copy(images)
    #         tt_correct += float((tt_pred.argmax(dim=1) == labels).sum())

    #     adv_images = adv_perturb(args, images, model_copy, labels)
    #     adv_pred = model_copy(adv_images)
    #     adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
    #     total += len(labels)

    #     if args.attack_type == 'aa':
    #         break 

    # return test_correct, adv_correct, total