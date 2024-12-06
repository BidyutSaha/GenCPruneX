import tensorflow as tf
from profiler import *
import numpy as np
from cprunex import *
from settings import *




with open("./Testdata/AIDER/stuffs_AIDER.pkl", 'rb') as file:
    print(file)
    loaded_data = pickle.load(file)

print(loaded_data)

# Load the model from the .h5 file
model = tf.keras.models.load_model('model_logs/best_model.h5')

# Load sample data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0


# Print the model summary to verify it's loaded correctly
model.summary()

input_shape = base_model.input_shape[1:]

x = 0

solution = [0.6,0,0.5,0.2]
new_model = custom_channel_prune(base_model,input_shape,solution,1)


new_model.summary()



ram , flash, mac = evaluate_hardware_requirements(new_model,(x_train, y_train))
print(f"Ram : {ram}, flash : {flash}, mac : {mac}")
