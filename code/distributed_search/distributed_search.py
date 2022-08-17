'''
Following the example from Ray Core that can be found here:
https://docs.ray.io/en/latest/ray-core/examples/plot_hyperparameter.html
'''

from math import inf
import random
import os
import csv
import click

import ray
#ray.init(address="auto")
ray.init()

# A function for generating random hyperparameters.
from generate_hp import generate_hyperparameters
# Predictor class encapsulates the performance predictor
from mlpf_predictor import Predictor_mlpf as Predictor
# Training function
from remote_fun import remote_fun


@ray.remote
def partial_train(config, n_epochs):
	return remote_fun(config, n_epochs)

'''
# intended to be used in cases were the trials that finish training continuing the partial training
# not used in current version
@ray.remote
def finish_training(trial, n_epochs):
	return
'''

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