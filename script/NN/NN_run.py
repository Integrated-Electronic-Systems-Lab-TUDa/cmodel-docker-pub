import sys
sys.path.insert(1,'../')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import NN_build_small
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm 
import NN_utility as NN_u
import transfer_func as transf
import utility_func as uf
import add_gaussian_noise
import time
import datetime 

import kennard_stone as ks
import prepare_data as data_source

# --- actuall traing of a NN model (is trained and saved with the method here)
def main():
    start_time = time.time()
    run_NN_new(save_path = './NN_models/Set6_NNlin_05', mdl_type = 'lin', epochs = 800,subtype_num = 3,random_seed = 7,use_other_data=True,
           es_patience = 75, es_delta = 2E-06,path_of_other_data = './data/set6_lat2_m13.tbl')
    print("--- Runtime:  %s seconds ---" % (time.time() - start_time))

    return
def run_NN(save_path = './NN_models/NN_test', mdl_type = 'lin', epochs = 100,subtype_num = None,
           random_seed = 26,use_other_data = False, path_of_other_data = './data/set6_lat2_m13.tbl',
           es_patience = 120, es_delta = 1E-07):

    yname = 'id'
    if use_other_data == True:
        scale_factor = (-1)*1E03
        all_data = pd.read_table(path_of_other_data)
        all_data.columns =["v1","v2","v3",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        voltage_colums = ['v1','v2','v3']
    else:
        scale_factor = 1E03
        all_data = pd.read_table("./data/id.tbl")
        all_data.columns =["v1","v2","v3","v4",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        voltage_colums = ['v1','v2','v3','v4']

    trainVal,test = uf.split_trainVal_test_set(all_data=all_data)
    
     # --- creating the validation dataset for this NN train run, can be different random seed every time
    train,val_data = train_test_split(trainVal,test_size=0.1,random_state=random_seed)

    if mdl_type == 'log':
        train,_ = transf.transf_Thung(train)
        val_data,_ = transf.transf_Thung(val_data)
        test,_ = transf.transf_Thung(test)
        yname = 'y_tilde'

    if subtype_num == 5 or subtype_num == 6:
        # --- in case want to train gaus noise augmented mdl
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 3 ,yname=yname)    
    elif subtype_num == 4 or subtype_num == 6:
        # --- in case want to train bound current mdl 
        train = uf.bound_current(train,yname,1E-15)
    elif subtype_num == 7:
        train = uf.bound_current(train,yname,1E-14)
    elif subtype_num == 8:
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
        num_copies = 3 ,yname=yname,sigma=1E-13)
    elif subtype_num == 9:
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
        num_copies = 6 ,yname=yname)
    elif subtype_num == 10:
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
        num_copies = 6 ,yname=yname,sigma=1E-13)

    print('--- Description of Train Set --- ')
    print(train.describe())

    X = train.loc[:,voltage_colums].to_numpy()
    y = train.loc[:,[yname]].to_numpy()

    #--- validation data is not really necessary for single run since we are not tuning any hyperparameter here
    X_val = val_data.loc[:,voltage_colums].to_numpy()
    y_val = val_data.loc[:,[yname]].to_numpy()

    if mdl_type == 'backend':
        model_path_log = './NN_models/NN_log_07'
        model_path_lin = './NN_models/NN_15'
        my_met = NN_u.R2
        mdl_lin = keras.models.load_model(model_path_lin,custom_objects = { 'R2': my_met})
        y_lin_pred = mdl_lin.predict(X)
        mdl_log= keras.models.load_model(model_path_log,custom_objects = { 'R2': my_met})
        sigma_log = mdl_log.predict(X)
        v1 = train.loc[:,['v1']]

        y_log_pred = transf.inv_transf_Thung(sigma_log,v1).to_numpy()
        previous_preds = pd.DataFrame(
        {'y_lin_pred': y_lin_pred.reshape(-1), 
        'y_log_pred': y_log_pred.reshape(-1),
        'sigma_log': sigma_log.reshape(-1)}
        )
        X = previous_preds.to_numpy()
     
    if subtype_num == 0:
        compiled_mdl = NN_build_small.build_backend(dim_in = 3, n_neurons = 10)
    else:
        compiled_mdl = NN_build_small.build_tanh_model(dim_in = 3, n_neurons = 32) 
    

    # ----- callbacks -----
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience,min_delta=es_delta,verbose=1,mode='min',restore_best_weights=True),tf.keras.callbacks.TensorBoard("./NN_tuning_res/RuntimeTest_logs")]
  

    # ---- running fit ----
    print('----- fitting the model-----')
    history = compiled_mdl.fit(X,y,batch_size = 100,epochs = epochs,callbacks = callbacks,
    validation_data=(X_val,y_val),verbose = 2)


    #----- saving the model ----
    compiled_mdl.save(save_path)


    # ---- printing metrics ----
    X_test = test.loc[:,voltage_colums].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()
    if mdl_type == 'backend':
        y_pred = mdl_lin.predict(X_test,batch_size = 100)
    else :
        y_pred = compiled_mdl.predict(X_test,batch_size = 100)
    print('--- '+save_path+' Metrics on test set for: ---')

    print('skm calc R2 score on test:'+str(skm.r2_score(y_true,y_pred)))
    print('np calc R2 score on test:'+str(NN_u.R2_np(y_true,y_pred)))

    loss,MAE,R2= compiled_mdl.evaluate(X_test,y_true,batch_size = 100)  # returns loss and metrics
    print("loss on test: %.8f" % loss)
    print("MAE on test: %.8f" % MAE)
    #print("MSE on test: %.8f" % MSE)
    print("keras calc R2 on test : %.8f" % R2)
    return

def run_NN_for_tuning(
    model = None,
    save_path = './NN_models/NN_test', mdl_type = 'linear',
    epochs = 100,subtype_num = None, random_seed = 26, yname = 'id',
    delta_early_stop = 1E-06,callback_tensorboard= None
    ):
    """ Similar to run_NN but with adjusted printout and more Method call Args. (Less internal Args)
    """
    compiled_mdl = model

    # --- Data Reading & Prepro
    scale_factor = 1E03
    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4",yname]
    all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
    trainVal,test = uf.split_trainVal_test_set(all_data=all_data)
    
    # --- creating the validation dataset for this NN train run, can be different random seed every time
    train,val_data = train_test_split(trainVal,test_size=0.1,random_state=random_seed)

    if mdl_type == 'log':
        train,_ = transf.transf_Thung(train)
        val_data,_ = transf.transf_Thung(val_data)
        test,_ = transf.transf_Thung(test)
        yname = 'y_tilde'

    # --- subtype_num defines what kind of preprocessing is performed
    if subtype_num == 5 or subtype_num == 6:
        # --- in case want to train gaus noise augmented mdl
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 3 ,yname=yname)    
    elif subtype_num == 4 or subtype_num == 6:
        # --- in case want to train bound current mdl 
        train = uf.bound_current(train,yname,1E-15)
    elif subtype_num == 7:
        train = uf.bound_current(train,yname,1E-14)
    elif subtype_num == 8:
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
        num_copies = 3 ,yname=yname,sigma=1E-13)
    elif subtype_num == 9:
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
        num_copies = 6 ,yname=yname)
    elif subtype_num == 10:
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
        num_copies = 6 ,yname=yname,sigma=1E-13)

    print(train.describe())

    X = train.loc[:,['v1','v2','v3','v4']].to_numpy()
    y = train.loc[:,[yname]].to_numpy()

    # --- validation data is of great importance when tuning -> metric for finding the best hyperparams

    X_val = val_data.loc[:,['v1','v2','v3','v4']].to_numpy()
    y_val = val_data.loc[:,[yname]].to_numpy()

    # --- actual training

    # ----- callbacks -----
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,min_delta=delta_early_stop,verbose=1),
    callback_tensorboard]

    # ---- running fit -----a
    print('----- fitting the model-----')
    print(X.shape)
    history = compiled_mdl.fit(X,y,batch_size = 100,epochs = epochs,validation_data=(X_val,y_val)
                            ,callbacks = [callback],verbose = 2)

    #----- saving the model ----
    # compiled_mdl.save(save_path)


    # ---- printing metrics ----
    # X_test = test.loc[:,['v1','v2','v3','v4']].to_numpy()
    # y_true = test.loc[:,[yname]].to_numpy()
    # if mdl_type == 'backend':
    #     y_pred = mdl_lin.predict(X_test,batch_size = 100)
    # else :
    y_pred = compiled_mdl.predict(X_val,batch_size = 100)

    # print('--- '+save_path+' Metrics on test set for: ---')
    print('skm calc R2 score on Val:'+str(skm.r2_score(y_val,y_pred)))
    # print('np calc R2 score on test:'+str(NN_u.R2_np(y_true,y_pred)))

    # loss,MAE,R2= compiled_mdl.evaluate(X_test,y_true,batch_size = 100)  # returns loss and metrics
    # print("loss on test: %.8f" % loss)
    # print("MAE on test: %.8f" % MAE)
    # #print("MSE on test: %.8f" % MSE)
    # print("keras calc R2 on test : %.8f" % R2)
    return history

def run_NN_new(save_path = '', mdl_type = 'lin', epochs = 100, augmentation_type = None,
           random_seed = 26, data_type = "DC", data_path = '',
           es_patience = 120, es_delta = 1E-07, _id = datetime.datetime.now().strftime("%Y-%m-%dT%H.M.%S")):

    voltage_colums = ["Vlat21", "Vfglat1", "Vtglat1"]

    if data_type == "PT":
        yname = 'id'
        scale_factor = (-1)*1E03
        all_data = pd.read_table(data_path)
        all_data.columns =["Vlat21", "Vfglat1", "Vtglat1",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)

    elif data_type == "DC":
        yname = 'ids'

        scale_factor = 1E03
        all_data = pd.read_json(data_path)
        # all_data.columns =["v1","v2","v3","v4",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)

    print('---- Original Data set: '+ (str(data_path))+'----')
    print(all_data.describe())

    
    trainVal,test = uf.split_trainVal_test_set(all_data=all_data)
    
     # --- creating the validation dataset for this NN train run, can be different random seed every time
    train,val_data = train_test_split(trainVal,test_size=0.1,random_state=random_seed)

    if augmentation_type == "gauss_noise_3" :
        # --- in case want to train gaus noise augmented mdl
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 3 ,yname=yname)    
    # elif subtype == 4 or subtype == 6:
    #     # --- in case want to train bound current mdl 
    #     train = uf.bound_current(train,yname,1E-15)
    # elif subtype == 7:
    #     train = uf.bound_current(train,yname,1E-14)
    # elif subtype == 8:
    #     train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
    #     num_copies = 3 ,yname=yname,sigma=1E-13)
    elif augmentation_type == "gauss_noise_6" :
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 6 ,yname=yname)
    # elif subtype == 10:
    #     train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
    #     num_copies = 6 ,yname=yname,sigma=1E-10)

    if mdl_type == 'log':
        #
        train,_ = transf.transf_Thung(data = train,yname = yname,improved_del='on')
        val_data,_ = transf.transf_Thung(data = val_data, yname = yname)
        test,_ = transf.transf_Thung(data = test,yname = yname)
        yname = 'y_tilde'
    

    print('--- Description of Train Set --- ')
    print(train.describe())

    X = train.loc[:,voltage_colums].to_numpy()
    y = train.loc[:,[yname]].to_numpy()

    #--- validation data is not really necessary for single run since we are not tuning any hyperparameter here
    X_val = val_data.loc[:,voltage_colums].to_numpy()
    y_val = val_data.loc[:,[yname]].to_numpy()

    if mdl_type == 'backend':
        model_path_log = './NN_models/NN_log_07'
        model_path_lin = './NN_models/NN_15'
        my_met = NN_u.R2
        mdl_lin = keras.models.load_model(model_path_lin,custom_objects = { 'R2': my_met})
        y_lin_pred = mdl_lin.predict(X)
        mdl_log= keras.models.load_model(model_path_log,custom_objects = { 'R2': my_met})
        sigma_log = mdl_log.predict(X)
        Vlat21 = train.loc[:,['Vlat21']]

        y_log_pred = transf.inv_transf_Thung(sigma_log,Vlat21).to_numpy()
        previous_preds = pd.DataFrame(
        {'y_lin_pred': y_lin_pred.reshape(-1), 
        'y_log_pred': y_log_pred.reshape(-1),
        'sigma_log': sigma_log.reshape(-1)}
        )
        X = previous_preds.to_numpy()
     
    if augmentation_type == 0:
        compiled_mdl = NN_build_small.build_backend(dim_in = 3, n_neurons = 10)
    else:
        compiled_mdl = NN_build_small.build_tanh_model(dim_in = 3, n_neurons = 32) 
    

    # ----- callbacks -----
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience,min_delta=es_delta,verbose=1,mode='min',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard("./NN_tuning_res/RuntimeTest_logs")]
  


    # ---- running fit ----
    print('----- fitting the model-----')
    history = compiled_mdl.fit(X,y,batch_size = 100,epochs = epochs,callbacks = callbacks,
    validation_data=(X_val,y_val),verbose = 2)


    #----- saving the model ----
    compiled_mdl.save(save_path + "/" + str(_id))


    # ---- printing metrics ----
    X_test = test.loc[:,voltage_colums].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()
    if mdl_type == 'backend':
        y_pred = mdl_lin.predict(X_test,batch_size = 100)
    else :
        y_pred = compiled_mdl.predict(X_test,batch_size = 100)
    print('--- '+save_path+' Metrics on test set for: ---')

    print('skm calc R2 score on test:'+str(skm.r2_score(y_true,y_pred)))
    print('np calc R2 score on test:'+str(NN_u.R2_np(y_true,y_pred)))

    loss,MAE,R2= compiled_mdl.evaluate(X_test,y_true,batch_size = 100)  # returns loss and metrics
    print("loss on test: %.8f" % loss)
    print("MAE on test: %.8f" % MAE)
    #print("MSE on test: %.8f" % MSE)
    print("keras calc R2 on test : %.8f" % R2)
    return compiled_mdl


def mx_run_NN_lin(save_path = '', epochs = 100, augmentation_type = None,
           random_seed = 26, data = None, Xlabels = ["Vlat21", "Vfglat1", "Vtglat1"], Ylabel = ["ids"],
           es_patience = 120, es_delta = 1E-07, _id = datetime.datetime.now().strftime("%Y-%m-%dT%H.M.%S"), bound_elat = None, n_neurons = 16, splitting_algorithm="TF", test_size = 0.15, shuffle=True, batch_size=400, dim_in = 3):

    

    print('---- Original Data set _id: '+ (str(_id))+'----')
    print(data.describe())

    if splitting_algorithm=="kennard_stone":

        trainVal = pd.DataFrame()
        test = pd.DataFrame()
        train = pd.DataFrame()
        val_data = pd.DataFrame()


        if test_size == 0.0:
            ks_x_trainVal = data[Xlabels]
            ks_y_trainVal = data[Ylabel]


        else:
            ks_x_trainVal, ks_x_test, ks_y_trainVal, ks_y_test = ks.train_test_split(data[Xlabels], data[Ylabel],test_size=test_size)
            test[Xlabels] = ks_x_test
            test[Ylabel] = ks_y_test
            
        trainVal[Xlabels] = ks_x_trainVal
        trainVal[Ylabel] = ks_y_trainVal

        # --- creating the validation dataset for this NN train run, can be different random seed every time
        ks_x_train, ks_x_valData, ks_y_train, ks_y_valData = ks.train_test_split(data[Xlabels], data[Ylabel],test_size=0.1)

        train[Xlabels] = ks_x_train
        train[Ylabel] = ks_y_train

        val_data[Xlabels] = ks_x_valData
        val_data[Ylabel] = ks_y_valData

        if test_size == 0.0:
            test[Xlabels] = val_data[Xlabels]
            test[Ylabel] = val_data[Ylabel]

    else:
        if test_size == 0.0:
            trainVal = data
        else:
            trainVal,test = uf.split_trainVal_test_set(all_data=data)

        # --- creating the validation dataset for this NN train run, can be different random seed every time
        train,val_data = train_test_split(trainVal,test_size=0.1,random_state=random_seed)

        if test_size == 0.0:
            test = val_data

    if augmentation_type == "gauss_noise_3" :
        # --- in case want to train gaus noise augmented mdl
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 3 ,yname=Ylabel[0])    
    # elif subtype == 4 or subtype == 6:
    #     # --- in case want to train bound current mdl 
    #     train = uf.bound_current(train,yname,1E-15)
    # elif subtype == 7:
    #     train = uf.bound_current(train,yname,1E-14)
    # elif subtype == 8:
    #     train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
    #     num_copies = 3 ,yname=yname,sigma=1E-13)
    elif augmentation_type == "gauss_noise_6" :
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 6 ,yname=Ylabel[0])
    # elif subtype == 10:
    #     train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
    #     num_copies = 6 ,yname=yname,sigma=1E-10)
    
    if bound_elat != None:
        train = uf.bound_current(train,Ylabel[0],bound_elat)


    print('--- Description of Train Set --- ')
    print(train.describe())

    X = train.loc[:,Xlabels].to_numpy()
    y = train.loc[:,Ylabel].to_numpy()

    #--- validation data is not really necessary for single run since we are not tuning any hyperparameter here
    X_val = val_data.loc[:,Xlabels].to_numpy()
    y_val = val_data.loc[:,Ylabel].to_numpy()

     
    compiled_mdl = NN_build_small.build_tanh_model(dim_in = dim_in, n_neurons = n_neurons) 
    

    # ----- callbacks -----
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience,min_delta=es_delta,verbose=1,mode='min',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard("./NN_tuning_res/RuntimeTest_logs")]
  


    # ---- running fit ----
    print('----- fitting the model-----')
    history = compiled_mdl.fit(X,y,
                               batch_size = batch_size,
                               epochs = epochs,
                               callbacks = callbacks,
                               validation_data=(X_val,y_val),
                               verbose = 2,
                               shuffle = shuffle)


    #----- saving the model ----
    compiled_mdl.save(save_path + "/" + str(_id), save_format="h5")


    # ---- printing metrics ----
    X_test = test.loc[:,Xlabels].to_numpy()
    y_true = test.loc[:,Ylabel].to_numpy()


    y_pred = compiled_mdl.predict(X_test,batch_size = 100)
    print('--- '+save_path+' Metrics on test set for: ---')
    
    R2 = skm.r2_score(y_true,y_pred)

    print('skm calc R2 score on test:'+str(skm.r2_score(y_true,y_pred)))
    print('np calc R2 score on test:'+str(NN_u.R2_np(y_true,y_pred)))

    loss,MAE= compiled_mdl.evaluate(X_test,y_true,batch_size = batch_size)  # returns loss and metrics

    sMAPE = uf.Smape(y_true, y_pred)

    real_R2, real_SMAPE = uf.evaluate_test_lin(compiled_mdl,test,Xlabels=Xlabels,Ylabel=Ylabel)

    print("loss on test: %.8f" % loss)
    print("MAE on test: %.8f" % MAE)
    #print("MSE on test: %.8f" % MSE)
    print("sMAPE : %.8f"  % sMAPE)
    print("keras calc R2 on test : %.8f" % R2)
    return compiled_mdl, loss,MAE,R2, sMAPE, X, y, real_R2, real_SMAPE




def mx_run_NN_log(save_path = '', epochs = 100, augmentation_type = None, sigma = 1e-14,
           random_seed = 26, data = None, Xlabels = ["Vlat21", "Vfglat1", "Vtglat1"], Ylabel = ["ids"],
           es_patience = 120, es_delta = 1E-07, _id = datetime.datetime.now().strftime("%Y-%m-%dT%H.M.%S"), Vds= "Vlat21", splitting_algorithm="TF", loss_func="MSE", n_neurons = 32, test_size=0.15, batch_size=400, dim_in = 3):


    print('---- Original Data set: '+ (str(_id))+'----')
    print(data.describe())

    if splitting_algorithm=="kennard_stone":

        trainVal = pd.DataFrame()
        test = pd.DataFrame()
        train = pd.DataFrame()
        val_data = pd.DataFrame()


        if test_size == 0.0:
            ks_x_trainVal = data[Xlabels]
            ks_y_trainVal = data[Ylabel]


        else:
            ks_x_trainVal, ks_x_test, ks_y_trainVal, ks_y_test = ks.train_test_split(data[Xlabels], data[Ylabel],test_size=test_size)
            test[Xlabels] = ks_x_test
            test[Ylabel] = ks_y_test
            
        trainVal[Xlabels] = ks_x_trainVal
        trainVal[Ylabel] = ks_y_trainVal

        # --- creating the validation dataset for this NN train run, can be different random seed every time
        ks_x_train, ks_x_valData, ks_y_train, ks_y_valData = ks.train_test_split(data[Xlabels], data[Ylabel],test_size=0.1)

        train[Xlabels] = ks_x_train
        train[Ylabel] = ks_y_train

        val_data[Xlabels] = ks_x_valData
        val_data[Ylabel] = ks_y_valData

        if test_size == 0.0:
            test[Xlabels] = val_data[Xlabels]
            test[Ylabel] = val_data[Ylabel]

    else:
        if test_size == 0.0:
            trainVal = data
        else:
            trainVal,test = uf.split_trainVal_test_set(all_data=data)

        # --- creating the validation dataset for this NN train run, can be different random seed every time
        train,val_data = train_test_split(trainVal,test_size=0.1,random_state=random_seed)

        if test_size == 0.0:
            test = val_data


    if augmentation_type == "gauss_noise_3" :
        # --- in case want to train gaus noise augmented mdl
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 3 ,yname=Ylabel[0],sigma=sigma)    
    # elif subtype == 4 or subtype == 6:
    #     # --- in case want to train bound current mdl 
    #     train = uf.bound_current(train,yname,1E-15)
    # elif subtype == 7:
    #     train = uf.bound_current(train,yname,1E-14)
    # elif subtype == 8:
    #     train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
    #     num_copies = 3 ,yname=yname,sigma=1E-13)
    elif augmentation_type == "gauss_noise_6" :
        train = add_gaussian_noise.augment_train_with_noise(trainVal = train, num_copies = 6 ,yname=Ylabel[0],sigma=sigma)
    # elif subtype == 10:
    #     train = add_gaussian_noise.augment_train_with_noise(trainVal = train, 
    #     num_copies = 6 ,yname=yname,sigma=1E-10)

    test_before_thung = test.copy()

    train,_ = transf.transf_Thung(data = train,Vds=Vds, yname = Ylabel[0],improved_del='on', epsilon=1E-3)
    val_data,_ = transf.transf_Thung(data = val_data,Vds=Vds, yname = Ylabel[0])
    test,_ = transf.transf_Thung(data = test,Vds=Vds,yname = Ylabel[0])
    yname = 'y_tilde'
    

    print('--- Description of Train Set --- ')
    print(train.describe())

    X = train.loc[:,Xlabels].to_numpy()
    y = train.loc[:,[yname]].to_numpy()

    #--- validation data is not really necessary for single run since we are not tuning any hyperparameter here
    X_val = val_data.loc[:,Xlabels].to_numpy()
    y_val = val_data.loc[:,[yname]].to_numpy()


    compiled_mdl = NN_build_small.build_tanh_model(dim_in = dim_in, n_neurons = n_neurons , loss_func= loss_func, name="NN_model_log") 
    

    # ----- callbacks -----
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience,min_delta=es_delta,verbose=1,mode='min',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard("./NN_tuning_res/RuntimeTest_logs")]
  


    # ---- running fit ----
    print('----- fitting the model-----')
    history = compiled_mdl.fit(X,y,batch_size = batch_size,epochs = epochs,callbacks = callbacks, validation_data=(X_val,y_val),verbose = 2)


    #----- saving the model ----
    compiled_mdl.save(save_path + "/" + str(_id))


    # ---- printing metrics ----
    X_test = test.loc[:,Xlabels].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()

    y_pred = compiled_mdl.predict(X_test,batch_size = 100)
    print('--- '+save_path+' Metrics on test set for: ---')

    R2 = skm.r2_score(y_true,y_pred)
    print('skm calc R2 score on test:'+str(skm.r2_score(y_true,y_pred)))
    print('np calc R2 score on test:'+str(NN_u.R2_np(y_true,y_pred)))

    loss,MAE= compiled_mdl.evaluate(X_test,y_true,batch_size = 100)  # returns loss and metrics

    sMAPE = uf.Smape(y_true, y_pred)

    print("loss on test: %.8f" % loss)
    print("MAE on test: %.8f" % MAE)
    #print("MSE on test: %.8f" % MSE)

    print("sMAPE : %.8f"  % sMAPE)

    print("keras calc R2 on test : %.8f" % R2)

    real_R2, real_SMAPE = uf.evaluate_test_log(compiled_mdl,test_before_thung,Xlabels=Xlabels)

    return compiled_mdl, loss,MAE,R2, sMAPE, real_R2, real_SMAPE



if __name__ == "__main__":
    main()

