#
# Basic linear regression using numpy, no pytorch
#
# Modeling sin(x) with a polynomial
# From older pytorch tutorial: 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#
import numpy as np
import math

# Load x and y data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Initialize parameters to random values
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
	# forward pass: compute predicted y
	# y = a + b *x + c * x ** 2 + d * x ** 3
	y_pred = a + b *x + c * x ** 2 + d * x ** 3

	# compute loss
	loss = np.square(y_pred - y).sum()
	if t % 100 == 99:
		print(t, loss)

	# Backprop to compute graidents with respect to loss
	grad_y_pred = 2.0 * (y_pred -y)
	grad_a = grad_y_pred.sum()
	grad_b = (grad_y_pred * x).sum()
	grad_c = (grad_y_pred * x**2).sum()
	grad_d = (grad_y_pred * x**3).sum()

	# update weights
	a -= learning_rate * grad_a
	b -= learning_rate * grad_b
	c -= learning_rate * grad_c
	d -= learning_rate * grad_d


print(f"result: y = {a} + {b} x + {c} x^2 + {d} x^3")

