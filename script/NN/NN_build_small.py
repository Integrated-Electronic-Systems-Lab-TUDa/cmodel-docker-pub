import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import NN_utility as NN_u

import utility_func as uf

#--- methods for building NN models (not trained, just defining structure and some hyperparams)
def build_simple_model(dim_in=4, n_neurons = 8, opimizer = 'Adam', name = "NN_model"):
    """Builds small Keras neural network model.

    Args:
        dim_in (int): Dimensionality of input. Will be 4
        n_neurons (int): Number of neurons in hidden layers.
        optimizer (str or Keras object, optional): Optimizer. Defaults to "adam".

    Returns:
        mdl (Keras object): Compiled model.
    """
    # ----building input layer----
    x = layers.Input(shape= (dim_in),name ="InputLayer")

    # ---- hidden layser ----
    zOne = layers.Dense(units=n_neurons, activation = "relu", name = "HiddenLayer1")(x)
    zTwo = layers.Dense(units=n_neurons, activation = "relu", name = "HiddenLayer2")(zOne)
    zThree = layers.Dense(units=n_neurons, activation = "relu", name = "HiddenLayer3")(zTwo)

    #output Layer
    y = layers.Dense(units = 1, activation = 'linear', name = "OutputLayer")(zThree)

    #build model
    mdl = keras.Model(inputs=x,outputs=y, name = name)
    
    print(mdl.summary())
    my_metrics = [keras.metrics.MeanAbsoluteError(),keras.metrics.MeanSquaredError(),NN_u.R2()]
    my_loss = keras.losses.MeanSquaredError()
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=2E-03),loss=my_loss,metrics = my_metrics)
    return mdl
    
def build_tanh_model(dim_in=4, n_neurons = 8, opimizer = 'Adam',lr=1.6E-03, loss_func="MSE", name="NN_model"):
    """Builds small Keras neural network model.

    Args:
        dim_in (int): Dimensionality of input. Will be 4
        n_neurons (int): Number of neurons in hidden layers.
        optimizer (str or Keras object, optional): Optimizer. Defaults to "adam".

    Returns:
        mdl (Keras object): Compiled model.
    """
    # ----building input layer----
    x = layers.Input(shape= (dim_in, ),name ="InputLayer")

    # ---- hidden layser ----
    zOne = layers.Dense(units=n_neurons, activation = "tanh", name = "HiddenLayer1")(x)
    zTwo = layers.Dense(units=n_neurons, activation = "tanh", name = "HiddenLayer2")(zOne)
    zThree = layers.Dense(units=n_neurons, activation = "tanh", name = "HiddenLayer3")(zTwo)

    #output Layer
    y = layers.Dense(units = 1, activation = 'linear', name = "OutputLayer")(zThree)

    #build model
    mdl = keras.Model(inputs=x,outputs=y, name = name)
    print(mdl.summary())

    # my_metrics = [keras.metrics.MeanAbsoluteError(),NN_u.R2]
    my_metrics = [keras.metrics.MeanAbsoluteError()]

    if loss_func == "MSE":
        my_loss = keras.losses.MeanSquaredError()
    elif loss_func == "decreasingMSE":
        my_loss = uf.decreasingMSE()
    elif loss_func == "sMAPE":
        my_loss = uf.loss_sMAPE()

    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=my_loss,metrics = my_metrics)
    return mdl

def build_tuning_model(dim_in=4, n_neurons = 8, opimizer = 'Adam', lr = 1E-03,
                        act_func = 'tanh', NN_mdl_name = 'NN_Model', 
                        my_loss = keras.losses.MeanSquaredError() 
                        ):
    """Builds small Keras neural network model.

    Args:
        dim_in (int): Dimensionality of input. Will be 4
        n_neurons (int): Number of neurons in hidden layers.
        optimizer (str or Keras object, optional): Optimizer. Defaults to "adam".
        lr(float): Learning rate
        my_loss(loss_func_obj): loss function used

    Returns:
        mdl (Keras object): Compiled model.
    """
    # ----building input layer----
    x = layers.Input(shape= (dim_in),name ="InputLayer")

    # ---- hidden layser ----
    zOne = layers.Dense(units=n_neurons, activation = act_func, name = "HiddenLayer1")(x)
    zTwo = layers.Dense(units=n_neurons, activation = act_func, name = "HiddenLayer2")(zOne)
    zThree = layers.Dense(units=n_neurons, activation = act_func, name = "HiddenLayer3")(zTwo)

    #output Layer
    y = layers.Dense(units = 1, activation = 'linear', name = "OutputLayer")(zThree)

    #build model
    mdl = keras.Model(inputs=x,outputs=y, name = NN_mdl_name)
    print(mdl.summary())

    my_metrics = [keras.metrics.MeanAbsoluteError(),NN_u.R2]
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=my_loss,metrics = my_metrics)
    return mdl

def build_backend(dim_in=3, n_neurons = 10, opimizer = 'Adam'):
    """Builds Backend  model

    Args:
        dim_in (int): Dimensionality of input. Will be 4
        n_neurons (int): Number of neurons in hidden layers.
        optimizer (str or Keras object, optional): Optimizer. Defaults to "adam".

    Returns:
        mdl (Keras object): Compiled model.
    """
    # ----building input layer----
    x = layers.Input(shape= (dim_in),name ="InputLayer")

    # ---- hidden layser ----
    zOne = layers.Dense(units=n_neurons, activation = "linear", name = "HiddenLayer1")(x)
    zTwo = layers.Dense(units=n_neurons, activation = "linear", name = "HiddenLayer2")(zOne)

    #output Layer
    y = layers.Dense(units = 1, activation = 'linear', name = "OutputLayer")(zTwo)

    #build model
    mdl = keras.Model(inputs=x,outputs=y, name = "NN_backend")
    print(mdl.summary())

    my_metrics = [keras.metrics.MeanAbsoluteError(),keras.metrics.MeanSquaredError(),NN_u.R2]
    my_loss = keras.losses.MeanSquaredError()
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=1E-03),loss=my_loss,metrics = my_metrics)
    return mdl
