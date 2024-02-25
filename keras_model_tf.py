"""
 @file   keras_model_py
 @brief  Script for keras tf model definition
 @author jsapas
"""

########################################################################
# import python-library
########################################################################
# from import
import tensorflow as tf
import tf_keras as keras

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Dense(128, input_dim=inputDim))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # Intermediate layers
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(8))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # Output layer
    model.add(keras.layers.Dense(inputDim))

    return model

def load_model(file_path):
    return keras.models.load_model(file_path)
