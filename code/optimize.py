from generate_data import sample_config, Cat_hp, Num_hp
import csv
from re import search
import joblib
import numpy as np
from trial import Trial
from sklearn.model_selection import train_test_split
from joblib import dump, load
import click
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

'''
WARNING SUBOPTIMAL SCRIPT: saves all trials and compares their predicted performance at the end.
HAS TO BE CHANGED TO: do generate a trial, partially train it, compare with the best and decide -> discard OR update best
'''

def train_n_confs(n_samples: int, train_ratio: float, search_space: dict, max_epochs: int, X, y):
	trials=[]
	accs = []
	losses = []
	for _ in range(n_samples):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_ratio))
		config = sample_config(search_space)
		print(config)
		trial = Trial(config, X_train, X_test, y_train, y_test)
		loss, acc = [],[]
		trial.run_n_epochs(max_epochs)
		trials.append(trial)
		accs.append(trial.acc[:])
		losses.append(trial.loss[:])
	return trials, np.array(accs), np.array(losses)

def finite_difs(curve):
    difs1 = []
    for j in range(1,curve.shape[0]):
        difs1.append(curve[j]-curve[j-1])
    difs2 = []
    for j in range(1,len(difs1)):
        difs2.append(difs1[j]-difs1[j-1])
    difs1 = np.array(difs1)
    difs2 = np.array(difs2)
    return difs1, difs2

def gen_x(config, curve):
	difs1, difs2 = finite_difs(np.array(curve))
	return np.append(np.append(np.append(np.array(list(config.values())),np.array(curve)),difs1),difs2)

def load_dataset():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X = np.concatenate([X_train, X_test]).astype('float32')/255.0 #normalize to 0-1 range
	y = to_categorical(np.concatenate([y_train, y_test]))
	return X, y

@click.command()
@click.option('--known_epochs', default=3, help='Number of epochs needed for the performance prediction.')
@click.option('--total_epochs', default=12, help='Number of epochs up to which the most promising configurations will be trained')
@click.option('--model_file', default="model.joblib", help='Joblib file storing a trained performance predictor')
@click.option('--scaler_file', default="scaler.joblib", help='Joblib file storing the X scaler used when trained the model')
@click.option('--top_k', default=10, help='Number of configurations that will be fully trained')
@click.option('--n_samples', default=100, help='Number of configurations that will be partially trained')

def main(known_epochs: int, total_epochs: int, model_file: str, scaler_file: str, top_k: int, n_samples: int):
	'''
	top_k = 10
	known_epochs = 3
	total_epochs = 12

	model_file = "modelo.joblib"
	x_scaler_file = "x_scaler_file"
	'''

	X, y = load_dataset()
	search_space = {}
	search_space["filters"] = Cat_hp([8, 16, 32, 64, 128])
	search_space["strides"] = Cat_hp([2, 3, 5])
	search_space["max_pool"] = Cat_hp([2, 3, 5])
	search_space["1st_dense"] = Cat_hp([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
	search_space["lr"] = Num_hp(type=float, lb= 1e-5, ub=0.1, log = True)
	search_space["momentum"] = Num_hp(type=float, lb= 0, ub=1)

	trials, _, curves = train_n_confs(n_samples = n_samples, train_ratio = 0.8, 
									search_space = search_space, max_epochs = known_epochs,
									X = X, y = y)
	
	#generate vecor X with features for each conf
	print(curves.shape)
	X = gen_x(trials[0].config, curves[0])
	X = np.stack((X,gen_x(trials[1].config, curves[0])))
	for i in range(2,len(trials)):
		print(X.shape)
		print(gen_x(trials[i].config, curves[i]).shape)
		X = np.vstack([X, gen_x(trials[i].config, curves[i])])

	#load pretrained model and scalers to do predictions
	model = load(model_file)
	x_scaler = load(scaler_file)

	X = x_scaler.transform(X)
	predicted = model.predict(X)

	#returns idx of top n values in array
	#use -predicted to find n smallest values
	best_idx = np.argpartition(-predicted, -top_k)[-top_k:]
	print(predicted)
	print(best_idx)
	target = []
	i = 0
	#only finish trainig for best configs
	for idx in best_idx.astype(int):
		trial = trials[idx]
		name = "trial_"+str(idx)
		trial.run_n_epochs(total_epochs-known_epochs)
		dump(trial, name+".joblib")
		i=i+1
		with open("results", 'a') as file:
				writer = csv.writer(file)
				writer.writerow([name+".csv", trial.acc[-1], trial.loss[-1]])

if __name__ == "__main__":
	main()

