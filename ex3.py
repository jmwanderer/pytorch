#
# Linear regression with tensors using autograd
#
# Modeling sin(x) with a polynomial
# From older pytorch tutorial: 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#
import torch
import math

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Input / output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
	# Forward pass: compute predicted y
	y_pred = a + b * x + c * x ** 2 + d * x** 3

	# Compute loss
	loss = (y_pred - y).pow(2).sum()
	if t % 100 == 99:
		print(t, loss.item())

	loss.backward()


	with torch.no_grad():
		a -= learning_rate * a.grad
		b -= learning_rate * b.grad
		c -= learning_rate * c.grad
		d -= learning_rate * d.grad

		a.grad = None
		b.grad = None
		c.grad = None
		d.grad = None


print(f"result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")

