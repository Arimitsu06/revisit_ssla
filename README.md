# Revisiting Self-Supervised Label Augmentation for Exemplar-Free Class-Incremental Learning (WORK IN PROGRESS)


## Implementation for Supervised Continual Learning

The code framework is taken from [PyCIL](https://github.com/G-U-N/PyCIL) and [LDC](https://github.com/alviur/ldc).

### Dependencies
1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)

### Datasets

We performed experiments for `CIFAR100`. When training on `CIFAR100`, this framework will automatically download it.

### Run experiments

The code for our SSLA method can be found in [models/lwf.py](https://github.com/arimitsu06/revisit_ssla/blob/main/models/lwf.py).

To use LDC with LwF, run

   ```
    python main.py --config=exps/lwf.json
   ```

To use LDC with SSLA and LwF, run

   ```
    python main.py --config=exps/ssla_ldc.json
   ```

### Configuration Options

```
ssla: [true, false]
ssla_tasks: ["init_only", "all"],
ssla_kd: ["nonaug_only", "all"],
rot90: [true, false],
freq_filter: [true, false],
freq_filter_radius: Int
```

The configs can be modified in [exps/ssla_lwf.json](https://github.com/arimitsu06/revisit_ssla/blob/main/exps/lwf.json).