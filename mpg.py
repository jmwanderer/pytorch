"""
Estimate MPG for various parameters of automobiles
"""
import torch
import math
import pandas as pd
import numpy as np

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
torch.set_default_device(device)

# Read dataset
def load_dataset():
	# Read dataset
	url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
	column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
	                'Acceleration', 'Model Year', 'Origin']
	raw_dataset = pd.read_csv(url, names=column_names,
	                          na_values='?', comment='\t',
	                          sep=' ', skipinitialspace=True)

	# Clean dataset
	dataset = raw_dataset.copy()
	dataset.isna().sum()
	dataset = dataset.dropna()
	dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
	dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
		
	# Split training and test data
	train_dataset = dataset.sample(frac=0.8, random_state=0)
	test_dataset = dataset.drop(train_dataset.index)
	train_features = train_dataset.copy()
	test_features = test_dataset.copy()
		
	# Split labels and features
	train_labels = train_features.pop('MPG')
	test_labels = test_features.pop('MPG')
	return train_features, train_labels


def train(model,x ,y, learning_rate):
	loss_fn = torch.nn.MSELoss(reduction='sum')

	optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
	for t in range(10000):
		# Forward pass: compute predicted y
		y_pred = model(x)

		# Compute loss
		loss = loss_fn(y_pred, y)
		if t % 100 == 99:
			print(t, loss.item())

		model.zero_grad()

		loss.backward()

		optimizer.step()

# Load data and normalize x values
x, y = load_dataset()
x = torch.tensor(x.to_numpy(dtype=np.float32))
y = torch.tensor(y.to_numpy(dtype=np.float32))
x = torch.nn.functional.normalize(x)

# Make a simple 1 layer linear model
model = torch.nn.Sequential(
	torch.nn.Linear(9,1),
	torch.nn.Flatten(0,1))
#train(model, x, y, 0.1)

# Make a 2 layer model
model = torch.nn.Sequential(
	torch.nn.Linear(9,64),
	torch.nn.ReLU(),
	torch.nn.Linear(64,1),
	torch.nn.Flatten(0,1))
train(model, x, y, 0.001)
