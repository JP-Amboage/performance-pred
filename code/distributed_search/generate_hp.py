from ray.tune import choice, uniform, quniform, loguniform, randint, qrandint

def generate_hyperparameters(**kwargs):
	return {
		# Optimizer parameters
		"lr": loguniform(1e-6, 3e-2).sample(),
		# "activation": "elu",
		# "batch_size_physical": quniform(16, 64, 16).sample(),
		# "batch_size_gun": quniform(100, 800, 100).sample(),
		# "batch_size_delphes": samp([8, 16, 24]).sample(),
		# "expdecay_decay_steps": quniform(10, 2000, 10).sample(),
		# "expdecay_decay_rate": uniform(0.9, 1).sample(),
		# Model parameters
		# "layernorm": quniform(0, 1, 1).sample(),
		# "ffn_dist_hidden_dim": quniform(64, 256, 64).sample(),
		# "ffn_dist_num_layers": quniform(1, 3, 1).sample(),
		# "distance_dim": quniform(32, 256, 32).sample(),
		# "num_node_messages": quniform(1, 3, 1).sample(),
		"num_graph_layers_id": quniform(0, 4, 1).sample(),
		"num_graph_layers_reg": quniform(0, 4, 1).sample(),
		"dropout": uniform(0.0, 0.5).sample(),
		"bin_size": choice([8, 16, 32, 64, 128]).sample(),
		# "clip_value_low": uniform(0.0, 0.2),
		# "dist_mult": uniform(0.01, 0.2),
		# "normalize_degrees": quniform(0, 1, 1),
		"output_dim": choice([8, 16, 32, 64, 128, 256]).sample(),
		# "lr_schedule": choice([None, "cosinedecay"])  # exponentialdecay, cosinedecay, onecycle, none
		"weight_decay": loguniform(1e-6, 1e-1).sample(),
	}
	
'''
#For simple mnist nn
def generate_hyperparameters(**kwargs):
	return {
		"filters": random.choice([8, 16, 32, 64, 128]),
		"strides": random.choice([2, 3, 5]),
		"max_pool": random.choice([2, 3, 5]),
		"1st_dense": random.choice([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]),
		"lr": np.exp(random.uniform(np.log(1e-5),np.log(0.1))),
		"momentum": random.uniform(0,1)
	}
'''