'''
Following the example from Ray Core that can be found here:
https://docs.ray.io/en/latest/ray-core/examples/plot_hyperparameter.html
'''

from math import inf
import random
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
from ray.tune import grid_search, choice, uniform, quniform, loguniform, randint, qrandint
#ray.init(address="auto")
ray.init()

# A function for generating random hyperparameters.
def generate_hyperparameters():
	return {
		# Optimizer parameters
		"lr": loguniform(1e-6, 3e-2).sample(),
		# "activation": "elu",
		# "batch_size_physical": quniform(16, 64, 16).sample(),
		# "batch_size_gun": quniform(100, 800, 100).sample(),
		# "batch_size_delphes": samp([8, 16, 24]).sample(),
		# "expdecay_decay_steps": quniform(10, 2000, 10).sample(),
		# "expdecay_decay_rate": uniform(0.9, 1).sample(),
		# Model parameters
		# "layernorm": quniform(0, 1, 1).sample(),
		# "ffn_dist_hidden_dim": quniform(64, 256, 64).sample(),
		# "ffn_dist_num_layers": quniform(1, 3, 1).sample(),
		# "distance_dim": quniform(32, 256, 32).sample(),
		# "num_node_messages": quniform(1, 3, 1).sample(),
		"num_graph_layers_id": quniform(0, 4, 1).sample(),
		"num_graph_layers_reg": quniform(0, 4, 1).sample(),
		"dropout": uniform(0.0, 0.5).sample(),
		"bin_size": choice([8, 16, 32, 64, 128]).sample(),
		# "clip_value_low": uniform(0.0, 0.2),
		# "dist_mult": uniform(0.01, 0.2),
		# "normalize_degrees": quniform(0, 1, 1),
		"output_dim": choice([8, 16, 32, 64, 128, 256]).sample(),
		# "lr_schedule": choice([None, "cosinedecay"])  # exponentialdecay, cosinedecay, onecycle, none
		"weight_decay": loguniform(1e-6, 1e-1).sample(),
	}
	'''
	For simple mnist nn
	return {
		"filters": random.choice([8, 16, 32, 64, 128]),
		"strides": random.choice([2, 3, 5]),
		"max_pool": random.choice([2, 3, 5]),
		"1st_dense": random.choice([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]),
		"lr": np.exp(random.uniform(np.log(1e-5),np.log(0.1))),
		"momentum": random.uniform(0,1)
	}
	'''

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
		return #like this for debugging
		self.svr = load(svr)
		self.x_scaler = load(x_scaler)
		self.y_scaler = load(y_scaler)

	def predict(self,config, curve):
		return curve[-1] #like this for debugging
		difs1, difs2 = self.__finite_difs(np.array(curve))
		X = np.append(np.append(np.append(np.array(list(config.values())),np.array(curve)),difs1),difs2).reshape(1, -1)
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
	#trial = Trial(config, X_train=train['X'], X_test=test['X'], y_train=train['y'], y_test=test['y'])
	trial = Trial(config,config_file_path = '../parameters/delphes.yaml')
	trial.run_n_epochs(n_epochs)
	return trial.loss

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
@click.option('--presults_file', default="presults.csv", help='File to store the results of partially trained configs')
@click.option('--fresults_file', default="fresults.csv", help='File to store the results of totally trained configs')

def main(known_epochs: int, total_epochs: int, model_file: str, x_scaler_file: str, y_scaler_file: str, top_k: int, n_samples: int,presults_file: str, fresults_file: str ):
	print(
        """	Starting run with:
        	known_epochs: {}
        	total_epochs: {}
        	model_file: {}
			x_scaler_file: {}
			y_scaler_file: {}
			top_k: {}
			n_samples: {}
			presults_file: {}
			fresults_file: {}
      	""".format(
            known_epochs,
            total_epochs,
            model_file,
            x_scaler_file,
			y_scaler_file,
			top_k,
			n_samples,
			presults_file,
			fresults_file
        )
    )

	#instantiate the performance predictor
	predictor = Predictor(svr=model_file, x_scaler = x_scaler_file, y_scaler = y_scaler_file)

	# Keep track of the best hyperparameters and the best predicted loss.
	best_trials = [None] * top_k
	best_pred = [inf] * top_k
	# A list holding the object refs for all of the experiments that we have
	# launched but have not yet been processed.
	remaining_ids = []
	hps_mapping = {}

	# Randomly generate sets of hyperparameters and launch a task to evaluate it.
	for i in range(n_samples):
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

		trial_loss = ray.get(result_id)
		hps = hps_mapping[result_id]

		pred = predictor.predict(hps, trial_loss)
		
		line = []
		for hp in list(hps.values()):
			line.append(hp)
		line.extend(trial_loss)
		#line.append(predictor.y_scaler.inverse_transform(pred.reshape(1, -1)))
		line.append(pred)
		print(pred)

		with open(presults_file, 'a') as file:
			writer = csv.writer(file)
			writer.writerow(line)
		
		if pred < best_pred[0]:
			best_pred = best_pred[1:] #remove the worst
			best_trials = best_trials[1:] #remove the worst
			flag_inserted = False
			for i in range(len(best_pred)):
				if(pred > best_pred[i]):
					best_pred.insert(i,pred)
					best_trials.insert(i,result_id)
					flag_inserted = True
					break
			if not flag_inserted:
				best_pred.insert(top_k-1,pred)
				best_trials.insert(top_k-1,result_id)
	
	#Launch tasks to finish evaluation of most promising trials
	for trial in best_trials:
		#trial_id = finish_training.remote(trial,total_epochs-known_epochs)
		trial_id = partial_train.remote(hps_mapping[trial],total_epochs)
		remaining_ids.append(trial_id)
		hps_mapping[trial_id] = hps_mapping.pop(trial)
	
	finished_trials = []
	i=0
	# Fetch and print the final results of the tasks in the order that they complete.
	while remaining_ids:
		# Use ray.wait to get the object ref of the first task that completes.
		done_ids, remaining_ids = ray.wait(remaining_ids)
		result_id = done_ids[0]
		trial_loss = ray.get(result_id)
		finished_trials.append(trial)
		#dump(trial, "trial_"+str(i)+".joblib")
		line = []
		for hp in list(hps_mapping[result_id].values()):
			line.append(hp)
		line.extend(trial_loss)
		with open(fresults_file, 'a') as file:
			writer = csv.writer(file)
			writer.writerow(line)
		i=i+1


if __name__ == "__main__":
	main()