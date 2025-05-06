#
# Linear regression with tensors using autograd and a neural network
#
# Modeling sin(x) with a polynomial
# From older pytorch tutorial: 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#
import torch
import math

# Input / output data
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Create an input set of data that is [ [ x, x**2, x**3], ... ]
p = torch.tensor([1,2,3])

# x = [ v1, v2, v3, ...]
# x.unsqueeze(-1) = [ [v1], [v2], [v3], ...]
# x.unsqueeze(-1).pow(p) = [ [v1, v1**2, v1**3], [v2, v1**2, v2**3], ...]
xx = x.unsqueeze(-1).pow(p)


# Build a linear regression model. Output of Linear is an array of arrays.
# Flatten to an array to match y data
model = torch.nn.Sequential(
	torch.nn.Linear(3,1),
	torch.nn.Flatten(0,1))

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
	# Forward pass: compute predicted y
	y_pred = model(xx)

	# Compute loss
	loss = loss_fn(y_pred, y)
	if t % 100 == 99:
		print(t, loss.item())

	model.zero_grad()

	loss.backward()


	with torch.no_grad():
		for param in model.parameters():
			param -= learning_rate * param.grad


linear_layer = model[0]

print(f"result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:,1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3")

