    # if args.precluster_method == 'kmeans':
    #     images, __ = next(iter(train_loader))
    #     patches = tokenembedder.splitter(images)
    #     patches = patches.view(-1, patches.size(-1))
        
    #     print('kmeans clustering fitting....')
    #     kmeans = KMeans(n_clusters=args.vocabulary_size, random_state=args.seed).fit(patches)
    #     kmeans_ids = kmeans.predict(patches)
    #     print('... done')
        
    #     patches = patches.to(args.device)
    #     kmeans_ids = torch.from_numpy(kmeans_ids).to(args.device).long()
    #     for __ in range(10):
    #         optimizer.zero_grad()
    #         output = tokenembedder.tokenizer(patches)
    #         loss = torch.nn.CrossEntropyLoss()(output, kmeans_ids)
    #         loss.backward()
    #         optimizer.step()
    #     correct, total = test_tokenizer(args, tokenembedder, test_loader)
    #     print(f'attacked tokenizer accuracy after kmeans: {correct/total:.4f}')