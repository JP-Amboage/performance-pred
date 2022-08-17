from gettext import npgettext
from joblib import load, dump
import numpy as np

class Predictor:
	def __init__(self, svr: str, x_scaler: str, y_scaler: str, **kwargs):
		#return #like this for debugging
		self.svr = load(svr)
		self.x_scaler = load(x_scaler)
		self.y_scaler = load(y_scaler)

	def predict(self,config, curve):
		#return curve[-1] #like this for debugging
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