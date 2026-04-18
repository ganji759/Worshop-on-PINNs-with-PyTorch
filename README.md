# PINNs Workshop (PyTorch)

This project contains a Jupyter Notebook implementation of a Physics-Informed Neural Network (PINN) in PyTorch for projectile motion under uniform gravity.

The model is trained with very few data points while also enforcing the governing physics equations.

## Overview

PINNs combine data-driven learning with physical constraints. Instead of fitting only observed points, the network is optimized to satisfy both:

- sparse trajectory observations
- the projectile motion ordinary differential equations

For this setup, the equations are:

$$
\frac{d^2x}{dt^2} = 0,\qquad \frac{d^2y}{dt^2} = g
$$

The network predicts horizontal position `x(t)` and vertical position `y(t)` over time.

## Requirements

- Python 3.8+
- PyTorch
- Matplotlib
- NumPy
- Pillow (for GIF creation)

Install dependencies:

```bash
pip install torch matplotlib numpy pillow
```

## Run

Launch the notebook:

```bash
jupyter notebook "Pinns workshop with PyTorch.ipynb"
```

## Notes

The trained PINN learns a trajectory close to the analytical solution while respecting physics constraints.

You can experiment with:

- gravity (`g`)
- initial velocity and launch angle
- network size
- number of training epochs