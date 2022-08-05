import numpy as np

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
