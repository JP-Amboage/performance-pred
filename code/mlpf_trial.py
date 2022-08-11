import tensorflow as tf

from raytune.search_space import set_raytune_search_parameters

from tfmodel.utils import get_strategy
from tfmodel.utils import get_datasets 
from tfmodel.utils import get_heptfds_dataset
from tfmodel.utils import get_lr_schedule
from tfmodel.utils import get_optimizer
from tfmodel.utils import set_config_loss
from tfmodel.utils import get_loss_dict
from tfmodel.utils import parse_config

from tfmodel.model_setup import make_model
from tfmodel.model_setup import prepare_callbacks
from tfmodel.model_setup import configure_model_weights
from tfmodel.model_setup import FlattenedCategoricalAccuracy

class Trial:
	def __init__(self, config, **kwargs):

		if 'config_file_path' in kwargs:
			config_file_path = kwargs['config_file_path']  
		if 'ntrain' in kwargs:
			ntrain = kwargs['ntrain']  
		if 'ntest' in kwargs:
			ntrain = kwargs['ntest']  
		if 'name' in kwargs:
			name = kwargs['name']  
		if 'seeds' in kwargs:
			name = kwargs['seeds']

		full_config, config_file_stem = parse_config(config_file_path)

		full_config = set_raytune_search_parameters(search_space=config, config=full_config)
		strategy, num_gpus = get_strategy()

		self.ds_train, self.num_train_steps = get_datasets(full_config["train_test_datasets"], full_config, num_gpus, "train")
		self.ds_test, self.num_test_steps = get_datasets(full_config["train_test_datasets"], full_config, num_gpus, "test")
		ds_val, ds_info = get_heptfds_dataset(full_config["validation_dataset"], full_config, num_gpus, "test", full_config["setup"]["num_events_validation"])

		total_steps = self.num_train_steps * full_config["setup"]["num_epochs"]
		'''
		callbacks = prepare_callbacks(
			full_config["callbacks"],
			ds_val,
			tune.get_trial_dir(),
			ds_info,
		)

		#callbacks = callbacks[:-1]  # remove the CustomCallback at the end of the list
		'''
		with strategy.scope():
			lr_schedule, optim_callbacks = get_lr_schedule(full_config, steps=total_steps)
			#callbacks.append(optim_callbacks)
			opt = get_optimizer(full_config, lr_schedule)

			self.model = make_model(full_config, dtype=tf.dtypes.float32)

			# Run model once to build the layers
			self.model.build((1, full_config["dataset"]["padded_num_elem_size"], full_config["dataset"]["num_input_features"]))

			full_config = set_config_loss(full_config, full_config["setup"]["trainable"])
			configure_model_weights(self.model, full_config["setup"]["trainable"])
			self.model.build((1, full_config["dataset"]["padded_num_elem_size"], full_config["dataset"]["num_input_features"]))

			loss_dict, loss_weights = get_loss_dict(full_config)
			self.model.compile(
				loss=loss_dict,
				optimizer=opt,
				sample_weight_mode="temporal",
				loss_weights=loss_weights,
				metrics={
					"cls": [
						FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
						FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
					]
				},
			)
			self.model.summary()

		self.loss=[]	
	def run_n_epochs(self, n: int):
		num_train_steps = self.num_train_steps
		num_test_steps = self.num_test_steps
		num_train_steps = 4 #for debugging on laptop
		num_test_steps = 4 #for debugging on laptop
		try:
			fit_result = self.model.fit(
				self.ds_train.repeat(),
				validation_data=self.ds_test.repeat(),
				epochs=n,
				steps_per_epoch=num_train_steps,
				validation_steps=num_test_steps,
			)
			self.loss.extend(fit_result.history['loss'])
		except tf.errors.ResourceExhaustedError:
			print("Error")



'''
# FOR DEUBUGGING 
from ray import tune
from ray.tune import grid_search, choice, uniform, quniform, loguniform, randint, qrandint

search_space = {
	# Optimizer parameters
	"lr": loguniform(1e-6, 3e-2),
	# "activation": "elu",
	# "batch_size_physical": quniform(16, 64, 16),
	# "batch_size_gun": quniform(100, 800, 100),
	# "batch_size_delphes": samp([8, 16, 24]),
	# "expdecay_decay_steps": quniform(10, 2000, 10),
	# "expdecay_decay_rate": uniform(0.9, 1),
	# Model parameters
	# "layernorm": quniform(0, 1, 1),
	# "ffn_dist_hidden_dim": quniform(64, 256, 64),
	# "ffn_dist_num_layers": quniform(1, 3, 1),
	# "distance_dim": quniform(32, 256, 32),
	# "num_node_messages": quniform(1, 3, 1),
	"num_graph_layers_id": quniform(0, 4, 1),
	"num_graph_layers_reg": quniform(0, 4, 1),
	"dropout": uniform(0.0, 0.5),
	"bin_size": choice([8, 16, 32, 64, 128]),
	# "clip_value_low": uniform(0.0, 0.2),
	# "dist_mult": uniform(0.01, 0.2),
	# "normalize_degrees": quniform(0, 1, 1),
	"output_dim": choice([8, 16, 32, 64, 128, 256]),
	# "lr_schedule": choice([None, "cosinedecay"])  # exponentialdecay, cosinedecay, onecycle, none
	"weight_decay": loguniform(1e-6, 1e-1),
}

if __name__ == "__main__":
	config = {}
	for k in search_space.keys():
		config[k] = search_space[k].sample()
	print(config)
	t = Trial(config=config,config_file_path = '../parameters/delphes.yaml')
	t.run_n_epochs(2)
	print(t.loss)
'''