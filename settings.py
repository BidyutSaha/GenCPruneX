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
base_model =  tf.keras.models.load_model("best_model.h5" )
base_model.summary()


# Load the data
loaded_train_data = np.load('./all_datasets/train_dataset_VWW.npz')
x_train = loaded_train_data['images']
y_train = loaded_train_data['labels']

loaded_val_data = np.load('./all_datasets/validation_dataset_VWW.npz')
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
constraints_specs= {"ram"   : 75*1024,    # 'ram' in bytes
                    "flash" : 100*1024,    # 'flash' in bytes
                    "macc"  : 1.5*100*1000 }



