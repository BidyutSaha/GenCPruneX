
import numpy as np 
import pandas as pd
import tensorflow as tf
from cprunex import *
from profiler import *
from util import *
import os



# To Load the Dataset

train_data = np.load('all_datasets/train_dataset_VWW.npz')
X = train_data['images']
Y = train_data['labels']

test_data = np.load('all_datasets/validation_dataset_VWW.npz')
X_test = test_data['images']
Y_test = test_data['labels']


# Load the Trained Model

base_model = tf.keras.models.load_model("best_model.h5" )

print("*****************Loadded Model********************* ")
base_model.summary()
ram_base, flash_base, mac_base = evaluate_hardware_requirements(base_model, (X,Y))
print(f"***************** [ Base Model | Resource FootPrints ] RAM {ram_base} Bytes, MACs {mac_base}, FLash {flash_base} Bytes")


# Execute Dynamic Pruning

num_pruneable_layers = model_to_chromosome_size(base_model)


ratios = [0.5, 0.9, 0.7, 0.2, 0.5, 0.3 ,0.1, 0.6 ,0.5, 0.9, 0.7, 0.9, 0.6, 0.7, 0.6, 0.4, 0.7, 0.9,
 0.8, 0.9, 0.6, 0.5, 0.3, 0.8, 0.3, 0.8, 0.4, 0.2, 0.9, 0.8, 0.3, 0.,  0.9, 0.9, 0.6, 0.2,
 0.4, 0.7, 0.8, 0.6, 0.9, 0.1, 0.1, 0.7, 0.8, 0.3, 0.8, 0.3, 0.7, 0.3, 0.,  0.9, 0.1, 0.2,
 0.7 , 0.1]

pruned_model= custom_channel_prune(base_model, input_shape=(96,96,3), pruning_ratios=ratios)


print("*****************Pruned Model********************** ")
pruned_model.summary()
ram_pruned, flash_pruned, mac_pruned = evaluate_hardware_requirements(pruned_model, (X,Y))
print(f"***************** [ Pruned Model | Resource FootPrints ] RAM {ram_pruned} Bytes, MACs {mac_pruned}, FLash {flash_pruned} Bytes")

print(f"****************  [ Effeciency   | Ram Reduction {100-(ram_pruned*100)/ram_base} %, Flash Reduction {100-(flash_pruned*100)/flash_base} %, MACs Reduction {100-(mac_pruned*100)/mac_base} %]")



# Fine-tune the Pruned Model to Reduce Reconstruction Loss

learning_rate = 0.001  
optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate)



pruned_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = pruned_model.fit(X, Y, epochs=50, batch_size=64, verbose=1 , shuffle=True , validation_data=(X_test, Y_test))
