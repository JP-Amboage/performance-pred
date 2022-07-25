'''
Following the example from Ray Core that can be found here:
https://docs.ray.io/en/latest/ray-core/examples/plot_hyperparameter.html
'''

from math import inf
import os
import csv
import click
import numpy as np
from filelock import FileLock
from trial import Trial
from joblib import load, dump

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import ray

ray.init()

# The number of sets of random hyperparameters to try.
num_evaluations = 10


# A function for generating random hyperparameters.
def generate_hyperparameters():
	return {
		"learning_rate": 10 ** np.random.uniform(-5, 1),
		"batch_size": np.random.randint(1, 100),
		"momentum": np.random.uniform(0, 1),
	}

def get_data_loaders(train_ratio = 0.8, random_state = 42):
	# We add FileLock here because multiple workers will want to
	# download data, and this may cause overwrites 
	with FileLock(os.path.expanduser("~/data.lock")):
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	X = np.concatenate([X_train, X_test]).astype('float32')/255.0 #normalize to 0-1 range
	y = to_categorical(np.concatenate([y_train, y_test]))
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_ratio),random_state=42)

	return {"X":X_train, "y":y_train}, {"X":X_test, "y":y_test}

class Predictor:
	def __init__(self, svr: str, x_scaler: str, y_scaler: str):
		self.svr = load(svr)
		self.x_scaler = load(x_scaler)
		self.y_scaler = load(y_scaler)

	def predict(self,config, curve):
		difs1, difs2 = self.__finite_difs(np.array(curve))
		X = np.append(np.append(np.append(np.array(list(config.values())),np.array(curve)),difs1),difs2)
		X = self.x_scaler.transform(X)
		return self.svr.predict(X)
	
	def __finite_difs(self,curve):
		difs1 = []
		for j in range(1,curve.shape[0]):
			difs1.append(curve[j]-curve[j-1])
		difs2 = []
		for j in range(1,len(difs1)):
			difs2.append(difs1[j]-difs1[j-1])
		difs1 = np.array(difs1)
		difs2 = np.array(difs2)
		return difs1, difs2

@ray.remote
def partial_train(config, n_epochs):
	train, test = get_data_loaders()
	trial = Trial(config, X_train=train['X'], X_test=test['X'], y_train=train['y'], y_test=test['y'])
	trial.run_n_epochs(n_epochs)
	return trial

@ray.remote
def finish_training(trial, n_epochs):
	trial.run_n_epochs(n_epochs)
	return trial

@click.command()
@click.option('--known_epochs', default=3, help='Number of epochs needed for the performance prediction.')
@click.option('--total_epochs', default=12, help='Number of epochs up to which the most promising configurations will be trained')
@click.option('--model_file', default="model.joblib", help='Joblib file storing a trained performance predictor')
@click.option('--x_scaler_file', default="x_scaler.joblib", help='Joblib file storing the X scaler used when trained the model')
@click.option('--y_scaler_file', default="y_scaler.joblib", help='Joblib file storing the y scaler used when trained the model')
@click.option('--top_k', default=10, help='Number of configurations that will be fully trained')
@click.option('--n_samples', default=100, help='Number of configurations that will be partially trained')
@click.option('--presults_file', default="presults.csv", help='File to store the results of partially rained configs')
def main(known_epochs: int, total_epochs: int, model_file: str, x_scaler_file: str, y_scaler_file: str, top_k: int, n_samples: int, presults_file: str ):
	
	#instantiate the performance predictor
	predictor = Predictor(svr=model_file, x_scaler = x_scaler_file, y_scaler = y_scaler_file)

	# Keep track of the best hyperparameters and the best predicted loss.
	best_trials = [None] * top_k
	best_pred = [inf] * top_k
	# A list holding the object refs for all of the experiments that we have
	# launched but have not yet been processed.
	remaining_ids = []
	# A dictionary mapping an experiment's object ref to its hyperparameters.
	# hyerparameters used for that experiment.
	hps_mapping = {}

	# Randomly generate sets of hyperparameters and launch a task to evaluate it.
	for i in range(num_evaluations):
		hps = generate_hyperparameters()
		trial_id = partial_train.remote(hps,known_epochs)
		remaining_ids.append(trial_id)
		hps_mapping[trial_id] = hps
	
	# Fetch and print the results of the tasks in the order that they complete.
	while remaining_ids:
		# Use ray.wait to get the object ref of the first task that completes.
		done_ids, remaining_ids = ray.wait(remaining_ids)
		# There is only one return result by default.
		result_id = done_ids[0]

		hps = hps_mapping[result_id]
		trial = ray.get(result_id)

		pred = predictor.predict(hps, trial.loss[:])

		line = []
		for hp in list(hps.values()):
			line.append(hp)
		line.extend(trial.loss)
		line.append(pred)
		with open("results", 'a') as file:
			writer = csv.writer(file)
			writer.writerow(line)
		
		if pred < best_pred[0]:
			best_pred = best_pred[1:] #remove the worst
			best_trials = best_trials[1:] #remove the worst
			for i in range(top_k-1):
				if(pred > best_pred[i]):
					idx = i if i < top_k-2 else top_k-1
					best_pred.insert(idx,pred)
					best_trials.insert(idx,trial)
				break

	#finish the training of the most promising configs
	print(#change to print to a file
		"""Best accuracy over {} trials was {:.3} with
		learning_rate: {:.2}
		batch_size: {}
		momentum: {:.2}
		""".format(
			num_evaluations,
			100 * best_accuracy,
			best_hyperparameters["learning_rate"],
			best_hyperparameters["batch_size"],
			best_hyperparameters["momentum"],
		)
	)

if __name__ == "__main__":
	main()