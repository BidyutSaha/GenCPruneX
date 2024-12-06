import tensorflow as tf
import numpy as np
from settings import *
import pickle
from tensorflow.keras.optimizers import Adam
from profiler import *
import warnings
warnings.filterwarnings("ignore")






class_weight = None



metric = 'val_accuracy'


# get the base model
base_model = tf.keras.models.load_model('./TestData/Visual_Wake_word/best_model.h5',)
base_model.summary()

# Load sample data

# Load train data
loaded_train_data = np.load('./TestData/Visual_Wake_word/train_dataset_VWW.npz')
x_train = loaded_train_data['images']
y_train = loaded_train_data['labels']

loaded_val_data = np.load('./TestData/Visual_Wake_word/validation_dataset_VWW.npz')
x_test = loaded_val_data['images']
y_test = loaded_val_data['labels']




ram, flash, macc = evaluate_hardware_requirements(base_model, (x_train, y_train))
base_specs = {"ram": ram, "flash": flash, "macc": macc}
print(base_specs)

# set epoch
EPOCH = 3

#set batch_size
BATCH_SIZE = 64



#set compilation setting
compilation_settings = {
    'optimizer':  Adam(learning_rate=0.001),
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}



#set permitable hardware specs
constraints_specs= {"ram"   : 1024*75 ,    # 'ram' in bytes
                    "flash" : 1024*64,    # 'flash' in bytes
                    "macc"  : 1000*60 }


if __name__ == "__main__":
    from cprunex import *

    
    ram, flash, macc = evaluate_hardware_requirements(base_model, (x_train, y_train))
    current_specs = {"ram": ram, "flash": flash, "macc": macc}
    print(current_specs)

    print("===========")

    new_model = uniform_channel_prune(base_model, base_model.input_shape[1:], pruning_ratio=0.9 , final_decision_layer_idx = len(base_model.layers)-1 , start_index = 1)
    new_model.summary()
    ram, flash, macc = evaluate_hardware_requirements(new_model, (x_train, y_train))
    current_specs = {"ram": ram, "flash": flash, "macc": macc}
    print(current_specs)



    
