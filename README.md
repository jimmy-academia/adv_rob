# efficient adversarial robustness

## experiment log

> insert new experiment log from the top

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