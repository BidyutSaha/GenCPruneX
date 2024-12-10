
import numpy as np 
import pandas as pd
import tensorflow as tf



# To Load the Dataset

train_data = np.load('all_datasets/train_dataset_VWW.npz')
X = train_data['images']
Y = train_data['labels']

test_data = np.load('all_datasets/validation_dataset_VWW.npz')
X_test = test_data['images']
Y_test = test_data['labels']


# Create a Model and Train it 


NUM_CLASSES = 2

# Define data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.15),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.3),
])

input_layer = tf.keras.layers.Input(shape=(96, 96, 3))
x = data_augmentation(input_layer)
base_model = tf.keras.applications.MobileNet(input_tensor = x, weights='imagenet', include_top=False, alpha=0.25, input_shape=(96, 96, 3))

# Unfreeze the base model layers
for layer in base_model.layers:
    layer.trainable = True



# Add a custom classification head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(.3)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(.3)(x)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Create the full model
model = tf.keras.models.Model(inputs=input_layer, outputs=output)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()


print(X.shape, Y.shape)
print(X_test.shape, Y_test.shape)


# Train the model
model.fit(X,Y,validation_data=(X_test,Y_test), epochs=50 , callbacks=[checkpoint_callback],class_weight=None, verbose=True, shuffle = True)


