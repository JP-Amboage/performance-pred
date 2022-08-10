import numpy as np
import random
from sklearn.model_selection import train_test_split

#calculate finite diferences of 1st and 2nd order
def finite_difs(curve):
	difs1 = []
	for i in range(curve.shape[0]):
		difs1.append([])
		for j in range(1,curve.shape[1]):
			difs1[i].append(curve[i][j]-curve[i][j-1])
	difs2 = []
	for i in range(curve.shape[0]):
		difs2.append([])
		for j in range(1,len(difs1[0])):
			difs2[i].append(difs1[i][j]-difs1[i][j-1])
	difs1 = np.array(difs1)
	difs2 = np.array(difs2)
	return difs1, difs2

#info of given dataframe
def get_df_info(data_name):
	df_info = {}
	if data_name == "mnist_4hp":
		df_info['df_path'] = "../data/mnist/all_4hp_rusty.csv"
		df_info['num_epochs'] = 12
		df_info['min_hp_idx'] = 0
		df_info['max_hp_idx'] = 3
		df_info['min_curve_idx'] = 16
	elif data_name == "mnist_6hp":
		df_info['df_path'] = "../data/mnist/all_6hp_rusty.csv"
		df_info['num_epochs'] = 12
		df_info['min_hp_idx'] = 0
		df_info['max_hp_idx'] =  5
		df_info['min_curve_idx'] = 18
	elif data_name == "mlpf":
		df_info['df_path'] = "../data/mlpf/delphes_trainings_processed.csv"
		df_info['num_epochs'] = 100
		df_info['min_hp_idx'] = 0
		df_info['max_hp_idx'] =  6
		df_info['min_curve_idx'] = 7
	
	return df_info

#custom crossvaidation for small training set with fixed test set
def small_train_r2_cv(model, X ,y , train_size: int, reps = 5, test_size = 0.5, rs = None):
	if rs == None:
		rs = random.randint(0,10000)

	if X.shape[0] * (1 - test_size) < train_size:
		train_size = int(X.shape[0] * (1 - test_size))
		print('WARNING: not enough data for this train_size, decrease test size or train size')
		print('\t---> Now using train size = ' + train_size)

	cvs = []
	for i in range(reps):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
		X_train = X_train[:train_size]
		y_train = y_train[:train_size]
		model.fit(X_train,y_train.ravel())
		cvs.append(model.score(X_test,y_test))

	return np.array(cvs), [rs + i for i in range(reps)]


def get_hps(df_info, df):
	hps = df[df.columns[df_info['min_hp_idx']:df_info['max_hp_idx']+1]].to_numpy()
	return hps

def get_curve(df_info, known_curve, df):
	curve = df[df.columns[df_info['min_curve_idx']:df_info['min_curve_idx']+int(df_info['num_epochs']*known_curve)]].to_numpy()
	return curve
	
def get_target(df_info, df):
	target = df[df.columns[df_info['min_curve_idx']+df_info['num_epochs']-2]].to_numpy()
	return target
