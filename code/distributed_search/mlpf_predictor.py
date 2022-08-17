from predictor import Predictor

class Predictor_mlpf(Predictor):
	def __init__(self, svr: str, x_scaler: str, y_scaler: str, **kwargs):
		return #like this for debugging

	def predict(self,config, curve):
		return curve[-1] #like this for debugging