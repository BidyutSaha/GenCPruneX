import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, 
    GlobalAveragePooling2D, Dense, Input, Flatten, MaxPooling2D
)
import numpy as np
import os

# Create subfolder for saving model and logs
log_dir = "model_logs"
os.makedirs(log_dir, exist_ok=True)

# Load sample data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

# Print dataset shapes
print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, Labels shape: {y_test.shape}")

def create_cnn_model_y(class_num=10):
    """
    Builds and compiles a CNN model for classification.
    
    Args:
        class_num (int): The number of output classes.
        
    Returns:
        model (tf.keras.Model): The compiled CNN model.
    """
    input_layer = Input(shape=(28, 28, 1))

    # First Conv Layer followed by MaxPooling
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', use_bias=False)(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second Conv Layer followed by MaxPooling
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', use_bias=False)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Dense Layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu', use_bias=False)(x)

    # Output Layer
    output = Dense(class_num, activation='softmax')(x)

    # Model creation and compilation
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

# Initialize model
model = create_cnn_model_y()

# Callbacks for saving the best model and logging
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(log_dir, 'best_model.h5'),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)


# Custom callback for logging training progress to a file
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        log_message = f"Epoch {epoch + 1}: " \
                      f"Loss = {logs['loss']:.4f}, " \
                      f"Accuracy = {logs['accuracy']:.4f}, " \
                      f"Val Loss = {logs['val_loss']:.4f}, " \
                      f"Val Accuracy = {logs['val_accuracy']:.4f}"
        print(log_message)  # Print to console
        with open(os.path.join(log_dir, "training_log.txt"), "a") as f:
            f.write(log_message + "\n")  # Append to log file

# Training the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=30,
    batch_size=64,
    verbose=1,
    callbacks=[checkpoint_callback, LoggingCallback()]
)

# Save best model
print(f"Best model saved at {os.path.join(log_dir, 'best_model.h5')}")
