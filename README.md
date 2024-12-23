# Efficient Adversarial Robustness

## Usage
- source 
```bash
cd src
mkdir cache
subl cache/rootdir # write the path to dataset download location
python main.py --model resnetcifar_zlqh --dataset cifar10 # for individual runs
python script/main_exp.py # script for full training and robustness evaluation
pyrhon script/qualitative.py # TODO: script for qualitative tests
```

## Experiment Log

> insert new experiment log from the top

### 2024-12-17
- db49 working on mnist, solid working on cifar10


### 2024-12-16
- TODO restart: step by step prepare setting => validation loss set early stopping
- dataset by dataset finalize!

### 2024-12-7
- reworked attack
- next: check `train_env/base.py`, `main_exp`, `train_env/ast.py` vs `attack.py` ==> esp. look at `pgd_attack`; coordinate so that pgd_attack can save iters, return 10, 20, 50 results together
==> save time by not always doing test inference!!!
- save evaluation results to json (dataset => model => train_env)

### 2024-12-6
- plan: train, save model; evaluate, save results
  1. train once, evaluate manytimes...
- finish AST v AT, and then do baseline #, 'MART', 'TRADES', 'FASTAT'...

### 2024-11-28
- finish conding the consolidation of afa, zlqh... into ablations of iptnet for model and train_env
- `todo: code logging and printing results; code the all script/experiment codes; run all exp script`

### 2024-11-27
- restart progress.

### 2024-11-20
- add usage msg to main.py; print when directly called (not from script/xx_exp.py)

### 2024-11-15
- reorganize location: README.md; src/; doc/; (exp/?)... maybe exp not good...
- configure src/main.py to test ablation between token and direct prediction for zlqhnet.high_predictor
  `python main.py` vs `python main.py --direct`

### 2024-11-14
- tokenization phase 1 high order embeeding token complete
- next: phase 2 = do tokenization work under adversarial noise?
- + restructure organize experiment code; plan writing. 

### 2024-11-13
> Back to work
- test if tokenization is meaningful... `cp exp/calc_lambda.py exp/calc_tokenization.py`

### 2024-10-29
- previously: finished afa trainer and network = average (zeroth order)+ fine tune
- today: finished zlqh network and trainer (except q) = 
  - zeroth order: 1 value is the average of entire patch
  - linear order: 3 values for width, height, and channel-wise direction gradient
  - [todo] quadratic order: 3 values
  - high order noise: T values for tokenization reconstruction into full patch

TODOs: quadratic order, exp/calc_lambda.py to calculalate lambda value for each order

### 2024-10-24
base = 0.2969 under AutoAttack
trying softmax better? => visualize reconstruction!

### 2024-10-23
(in screen run) => run 75 epoch, look if adv_acc go up and then down
next steps: step size scheduling (use ttt), ??


### 2024-10-02

breakthrough this week!!!
- -> running AT cifar10 mobilenet @solid [firsttime] --> up to 0.35 adv acc
- -> write/run AST cifar10 mobileapt patch 1

### 2024-09-30

[plan]
- follow chatgpt and initialize code
- test run 
- start initial full run
[result]
- ...
[TODO]
- ...

### 2024-09-23
- initialize new repository

[TODO]
1. basic modualization... a) interchange training method (ipt and other at, quantization methods) vs models vs datasets vs attacks
2. setup list of models and start 

## Notes

Evaluating the adversarial robustness of adaptive test-time defenses
Uncovering Adversarial Risks of Test-Time Adaptation

## Future Todo:
- check utils out of function global variables
- test time training
- cw attack