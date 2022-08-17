from mlpf_trial import Trial

def remote_fun(config, n_epochs, **kwargs):
	#should be rewritten without using Trial 
	trial = Trial(config,config_file_path = '../parameters/delphes.yaml')
	trial.run_n_epochs(n_epochs)
	return trial.loss


'''
# Example for mnist were we need aux functions for the remote_fun
# MUST IMPORT A DIFFERENT TRIAL!!!!
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from filelock import FileLock

def get_data_loaders(train_ratio = 0.8, random_state = 42):
	# We add FileLock here because multiple workers will want to
	# download data, and this may cause overwrites 
	with FileLock(os.path.expanduser("~/data.lock")):
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	X = np.concatenate([X_train, X_test]).astype('float32')/255.0 #normalize to 0-1 range
	y = to_categorical(np.concatenate([y_train, y_test]))
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_ratio),random_state=42)

	return {"X":X_train, "y":y_train}, {"X":X_test, "y":y_test}


def remote_fun(config, n_epochs, **kwargs):
	train, test = get_data_loaders()
	trial = Trial(config, X_train=train['X'], X_test=test['X'], y_train=train['y'], y_test=test['y'])
'''