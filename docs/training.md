# Training

I wrote down the effect sizes, metrics, sample counts and selection rule before fitting a model. [Experiment protocol](protocol.md)

I train the one-frame model and temporal head on the same 240 clips. Still frames outnumber any one direction, so each class receives weight proportional to the inverse of its training-frame frequency. I use all 80 frame targets in every clip. Adam uses a fixed learning rate of 0.003, minibatches of 32 whole clips, seed 773 and 140 epochs. Both heads receive the same minibatch permutation per epoch. Training and validation losses in the curves are full-split evaluations after each epoch.

I choose each checkpoint independently by minimum validation weighted cross-entropy and save a state dictionary. The test split enters the evaluation command after both fits. I use deterministic PyTorch operations, set the cuBLAS workspace configuration and flatten batch/time for deterministic CUDA cross-entropy. The local run records model size, epoch, elapsed training seconds and peak live PyTorch allocation for each fit. That memory measure excludes the driver, display and CUDA context. [Training record](../artifacts/training.json)

The shared handcrafted encoder has zero learned parameters. The frame head is a per-sample MLP implemented with pointwise convolutions. The temporal head adds normalized neighborhood offsets. Doubling down on a large pretrained vision encoder would make visual appearance a second experimental variable and require unrelated weights. A learned encoder becomes more relevant with multiple real subjects and backgrounds.

See [reproduction](reproduction.md) for commands, [math](math.md) for equations and [evaluation](evaluation.md) for measurements.
