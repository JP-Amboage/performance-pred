from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

class Trial:
	def __init__(self, config, X_train, X_test, y_train, y_test) -> None:
		self.config={}
		self.config["filters"] = config["filters"] if "filters" in config else 32
		self.config["strides"] = config["strides"] if "strides" in config else 3
		self.config["max_pool"] = config["max_pool"] if "max_pool" in config else 2
		self.config["1st_dense"] = config["1st_dense"] if "1st_dense" in config else 100
		self.config["lr"] = config["lr"] if "lr" in config else 0.01
		self.config["momentum"] = config["momentum"] if "momentum" in config else 0.9


		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		self.model = Sequential()
		self.model.add(Conv2D(self.config["filters"], (self.config["strides"], self.config["strides"]), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		self.model.add(MaxPooling2D((self.config["max_pool"], self.config["max_pool"])))
		self.model.add(Flatten())
		self.model.add(Dense(self.config["1st_dense"], activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(10, activation='softmax'))
		
		# compile model
		opt = SGD(learning_rate=self.config["lr"], momentum=self.config["momentum"])
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
	def run_n_epochs(self, n: int):
		self.model.fit(self.X_train, self.y_train, epochs=n, batch_size=32)
		self.loss, self.acc =  self.model.evaluate(self.X_test, self.y_test, verbose=0)
