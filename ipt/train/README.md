## two_stage
> two stage process to prepare model

### method
unsupervised traing tokenization by 
    - 1) adversarial robustness
    - 2) similarity to anchor vectors
supervised training classifier (use anchor for embedding)
### result
works well against pgd attack
fails against square attack

## iter_auto
> iterative pepare tokenizer by autoencoder and square attack robustness 

### method

[The tokenizer training]
- autoencoder
<pre>
repeat N_auto:
    image -- tokenizer --> [tokens] == embedding -- decoder --> image
    [tokens] == learnable embedding (softmax with increasing temperature)
</pre>

- square attack
<pre>
repeat N_attack:
    full image perturbation (aim for all token modification)
        with iteration increasing budget 

    adversarial training for tokenizer with perturbed patches
</pre>

[Full Model]
<pre>
- repeat epochs:
    [tokenizer traning]
    check tokenizer on test set
    train classifier
    test full model
</pre>
