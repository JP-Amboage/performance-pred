from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

class Trial:
	def __init__(self, config, **kwargs):
	
		self.config=config
		self.X_train = kwargs['X_train']
		self.X_test = kwargs['X_test']
		self.y_train = kwargs['y_train']
		self.y_test = kwargs['y_test']
		self.loss = []
		self.acc = []
		self.model = Sequential()

		hps={}
		hps["filters"] = config["filters"] if "filters" in config else 32
		hps["strides"] = config["strides"] if "strides" in config else 3
		hps["max_pool"] = config["max_pool"] if "max_pool" in config else 2
		hps["1st_dense"] = config["1st_dense"] if "1st_dense" in config else 100
		hps["lr"] = config["lr"] if "lr" in config else 0.01
		hps["momentum"] = config["momentum"] if "momentum" in config else 0.9
		self.model.add(Conv2D(hps["filters"], (hps["strides"], hps["strides"]), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		self.model.add(MaxPooling2D((hps["max_pool"], hps["max_pool"])))
		self.model.add(Flatten())
		self.model.add(Dense(hps["1st_dense"], activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(10, activation='softmax'))
		
		# compile model
		opt = SGD(learning_rate=hps["lr"], momentum=hps["momentum"])
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
	def run_n_epochs(self, n: int):
		for _ in range(n): #done this way to be able to store the learning curve
			self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32)
			loss, acc =  self.model.evaluate(self.X_test, self.y_test, verbose=0)
			self.loss.append(loss)
			self.acc.append(acc)
