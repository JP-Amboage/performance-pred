import csv
import random
import numpy as np
from trace import Trace
from os.path import exists
from trial import Trial
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import click


# load train and test dataset
def load_dataset():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X = np.concatenate([X_train, X_test]).astype('float32')/255.0 #normalize to 0-1 range
	y = to_categorical(np.concatenate([y_train, y_test]))
	return X, y

class Hyperparam:
	def __init__(self):
		pass
	def get_random(self):
		pass

class Cat_hp(Hyperparam):
	def __init__(self, categories):
		self.categories = categories
	def get_random(self):
		return random.choice(self.categories)

class Num_hp(Hyperparam):
	def __init__(self, type, lb, ub, log = False):
		self.type = type
		self.lb = lb
		self.ub = ub
		self.log = log
	def get_random(self):
		if self.type == int:
			return random.randrange(self.lb,self.ub)
		if self.type == float:
			if(self.log):
				return np.exp(random.uniform(np.log(self.lb),np.log(self.ub)))
			return random.uniform(self.lb,self.ub)


def sample_config(search_space):
	config = {}
	for hp_name in search_space.keys():
		config[hp_name] = search_space[hp_name].get_random()
	return config

@click.command()
@click.option('--n_samples', default=0, help='Number of random samples to generate.')
@click.option('--max_epochs', default=12, help='Number of epochs.')
@click.option('--filename', default="dataset.csv", help='CSV file where the generated samples will be stored')
@click.option('--train_ratio', default=0.8, help='Train ratio used')
def main(n_samples: int, max_epochs: int, filename :str, train_ratio: float):

	#n_samples = 2
	#max_epochs = 12
	#filename = "test.csv"
	#train_ratio = 0.8

	search_space = {}
	search_space["filters"] = Cat_hp([8, 16, 32, 64, 128])
	search_space["strides"] = Cat_hp([2, 3, 5])
	search_space["max_pool"] = Cat_hp([2, 3, 5])
	search_space["1st_dense"] = Cat_hp([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
	search_space["lr"] = Num_hp(type=float, lb= 1e-5, ub=0.1, log = True)
	search_space["momentum"] = Num_hp(type=float, lb= 0, ub=1)

	if not exists(filename):
		line = list(search_space.keys())
		line.extend(["acc_"+str(i) for i in range(max_epochs)])
		line.extend(["loss_"+str(i) for i in range(max_epochs)])
		with open(filename, 'a') as file:
			writer = csv.writer(file)
			writer.writerow(line)

	X, y = load_dataset()

	for _ in range(n_samples):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_ratio))
		config = sample_config(search_space)
		print(config)
		trial = Trial(config, X_train, X_test, y_train, y_test)
		loss, acc = [],[]
		for __ in range(max_epochs):
			trial.run_n_epochs(1)
			loss.append(trial.loss)
			acc.append(trial.acc)
		line = list(trial.config.values())
		line.extend(acc)
		line.extend(loss)
		with open(filename, 'a') as file:
			writer = csv.writer(file)
			writer.writerow(line)


if __name__ == "__main__":
	main()