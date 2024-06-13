# Image patch tokenization solves adversarial robustness
> complete redevelopment 


## repo structure:
- `config.py`: to process configurations
- `ipt/`: code modules
    - `data.py`: load dataset, form dataloaders
    - `networks/`: model architecture designs
    - `attacks/`: adversarial attack functions
    - `train/`: training procedures
    - `utils.py`: utility functions
- `dev/`: experiment scripts
    - e.g. `basic/`
        - `mnist.py`
        - `mnist1.yaml`
        - `test2.yaml`
- `bin/`: code backups (to be removed)
- `(cache/)`: (git ignore) store info i.e. rootdir
- `(ckpt/)`: (git ignore) store checkpoints
    - e.g. `basic/`
        - e.g. `test2/`

usage example: `python dev/basic/mnist.py -c dev/basic/test2.yaml`

## Current Agenda:

- test black box and other attacks