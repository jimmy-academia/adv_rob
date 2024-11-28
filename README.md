# Efficient Adversarial Robustness

## Usage
- source 
```bash
cd src
python main.py --model resnetcifar_zlqh --dataset cifar10 # for individual runs
python script/main_exp.py # script for full training and robustness evaluation
pyrhon script/qualitative.py # TODO: script for qualitative tests
```


## Experiment Log

> insert new experiment log from the top

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