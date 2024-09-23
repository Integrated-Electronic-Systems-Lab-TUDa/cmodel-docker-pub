from keras import backend as kb
import numpy as np
import tensorflow as tf
import backend_ensembled

# --- utility method for NN models.  
def R2 (y_true, y_pred):
    SS_res =  kb.sum(kb.square( y_true-y_pred ))
    SS_tot = kb.sum(kb.square( y_true - kb.mean(y_true) ) )
    # initaly there was: kb.epsion = 1e-07 added to SS_tot to ensure it is != 0 - >  will make no differce 
    return ( 1 - SS_res/(SS_tot+kb.epsilon()))

def R2_np (y_true, y_pred):
    SS_res =  np.sum(( y_true-y_pred )**2)
    SS_tot = np.sum(( y_true - np.mean(y_true) )**2 )
    # initaly there was: K.epsion = 1e-07 added to SS_tot to ensure it is != 0 - >  will make no differce 
    return ( 1 - SS_res/(SS_tot))

def my_loss_MSE(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def my_loss_MSE_and_PE(y_true, y_pred):
    alpha = 0
    beta = 1E-5
    
    # --- MSE calc ----
    MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))

    # --- Percentage Error calc ---
    AE  = tf.math.abs(y_pred - y_true)
    epsilon = tf.constant([1E-20])
    PE = tf.math.abs(tf.divide(AE,tf.maximum(epsilon,tf.math.abs(y_true))))
    # --- mean of PE with limitation to 3000% as Mean absolute caped percentage Error ----
    PE_trsh = tf.constant([30.00E10])
    MACPE = tf.reduce_mean(tf.minimum(PE_trsh, PE))
    loss = alpha * MSE +beta * MACPE
    return loss

def my_loss_Mp4E(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(tf.square(y_true - y_pred)))

def i_lin_log_backend(y_pred_lin,test_data,sigma_log):
    i_pred = backend_ensembled.ensembled_backend(y_pred_lin,test_data,sigma_log)
    return i_pred
