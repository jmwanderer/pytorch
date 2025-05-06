#
# Basic linear regression with tensors
#
# Modeling sin(x) with a polynomial
# From older pytorch tutorial: 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#
import torch
import math

# Try setting the default type and compute device
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

# Input / output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
	# Forward pass: compute predicted y
	y_pred = a + b * x + c * x ** 2 + d * x** 3

	# Compute loss
	loss = (y_pred - y).pow(2).sum().item()
	if t % 100 == 99:
		print(t, loss)

	# Backprop
	grad_y_pred = 2.0 * (y_pred - y)
	grad_a = grad_y_pred.sum()
	grad_b = (grad_y_pred * x).sum()
	grad_c = (grad_y_pred * x ** 2).sum()
	grad_d = (grad_y_pred * x ** 3).sum()

	a -= learning_rate * grad_a
	b -= learning_rate * grad_b
	c -= learning_rate * grad_c
	d -= learning_rate * grad_d


print(f"result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")

