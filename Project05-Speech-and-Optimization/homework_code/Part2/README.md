# Homework 5 - Part 2: Optimization (SGD, Adam, RMSProp)

This project implements and compares three optimization algorithms
(SGD, Adam, RMSProp) on the 2D Rosenbrock function, following the
assignment instructions.

Note: this assignment has no dataset and no trained neural network, so
the `data/`, `models/`, and `notebooks/` folders from the course's
standard project template were left out. Everything needed fits in
`scripts/`, `configs/`, `utils/`, and `results/`.

## Folder structure

```
homework_code/
├── scripts/
│   ├── rosenbrock.py   # Part 1: rosenbrock(), grad_x(), grad_y()
│   ├── optimizers.py   # Part 2: SGD, Adam, RMSProp (subclasses of torch.optim.Optimizer)
│   ├── train.py         # runs one optimizer until gradient norm < threshold
│   ├── evaluate.py      # simple summary metrics for a run
│   └── main.py           # runs every part of the assignment (1, 3, 4, 5, 6)
├── configs/
│   └── config.yaml       # learning rates, starting points, threshold, noise std
├── utils/
│   ├── plot_utils.py     # all plotting functions
│   └── metrics.py        # Euclidean distance to the optimum
├── results/              # plots and results_summary.json are saved here   
├── README.md
└── .gitignore
```

## Requirements

- Python 3.9+
- torch
- numpy
- matplotlib
- pyyaml

Install with:

```bash
pip install torch numpy matplotlib pyyaml
```

## How to run

From the `homework_code` folder:

```bash
python scripts/main.py
```

This will:

1. Plot the Rosenbrock surface on `[-1, 1] x [-1, 1]` (Part 1).
2. Run SGD, Adam, and RMSProp from `(x0, y0) = (-1, -1)` at learning
   rate 0.1, continuing until the gradient norm drops below 0.001
   (Part 2, item 3). Plot the function value, distance to the optimum,
   and the 3D trajectory.
3. Repeat step 2 at learning rates 0.01 and 1, and create combined
   comparison plots showing all three rates (0.1, 0.01, 1) for each
   optimizer (Part 2, item 4).
4. Repeat step 2 from three different starting points
   (Part 2, item 5).
5. Repeat step 2 with small Gaussian noise added to the gradients
   (Part 2, item 6).

All figures are saved as `.png` files in `results/`, and a
`results_summary.json` file collects the final loss, final distance,
final gradient norm, epochs run, and convergence epoch (if reached)
for every run.

## Notes on the implementation

- Gradients are computed by hand with `grad_x` / `grad_y` (Part 1),
  not with PyTorch autograd, since the assignment asks for them to be
  derived manually.
- The three optimizers subclass `torch.optim.Optimizer` and implement
  the `step(closure=None)` method following the standard PyTorch
  optimizer interface. Each optimizer maintains its internal state
  (moving averages for Adam and RMSProp) in the per-parameter `state`
  dict provided by the base class.
- The training loop runs **until convergence** — defined as the L2 norm
  of the gradient dropping below 0.001 — rather than for a fixed
  number of epochs. A `num_epochs` safety cap (default 10000) prevents
  infinite loops when an optimizer diverges.
- The default (baseline) learning rate is 0.1, as required by the
  assignment.
- All absolute file paths are avoided; every path in the code is built
  relative to the location of the script itself.
