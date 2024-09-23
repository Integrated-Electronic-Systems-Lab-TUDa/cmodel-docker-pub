#import sys
#sys.path.insert(1,'c:\\Users\\jowil\\OneDrive\\Dokumente\\Uni\\MSc_4_Semester\\020_python\\021_symbolic_reg')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm 
import utility_func as uf
import NN.NN_utility as NN_u
import transfer_func as transf
import visualize_results as visres
# --- method for analysing NN models
def main():
    #analyse_singel(model_path='./NN_models/Set7_NNlog_03',type = 'log',test_all=True,
    #               use_other_data='ds3',modle_inputs=['v1','v2','v3'],set_v4_null=True)
    analyse_ensembled(backend_type='simple',model_path_log=name_list_NN('NNs6')[1],
    model_path_lin=name_list_NN('NNs6')[0],test_all=True, use_other_data='ds3',modle_inputs=['v1','v2','v3'],
    set_v4_null=True)
    return

def analyse_singel(
    model_path = './NN_models/NN_both_01',
    type = 'linear',
    test_all = False,
    use_other_data = 'off',
    path_of_other_data ="./data/set6_lat2_m13.tbl",
    modle_inputs = ['v1','v2','v3','v4'],
    set_v4_null = False
):
    # --- loading non standart data ---
    
    if use_other_data == 'ds3':
        yname = 'i_lat'
        scale_factor = -1E03
        all_data = pd.read_table(path_of_other_data)
        all_data.columns =["v1","v2","v3",yname]
        all_data['v4'] = np.zeros(shape =(len(all_data['v3']),1))
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        test_all = True
    elif use_other_data == 'ds5':
        yname = 'i_lat'
        scale_factor = -1E03
        all_data = pd.read_csv("./data/Set5_data.csv")
        all_data.columns =["v1","v2","v3","v4",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        test_all = True
    # --- loading the standart data Dataset ---
    else:
        yname = 'i_lat'
        scale_factor = 1E03
        all_data = pd.read_table("./data/id.tbl")
        
        all_data.columns =["v1","v2","v3","v4",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        if set_v4_null:
                voltages =  {
                'v1' : None,
                'v2' : None,
                'v3' : None,
                'v4' : 0.0 }
                all_data = visres.select_data_one_const(voltages=voltages,df_full=all_data,epsilon=0.001)
        

        # --- using only unseen data for testing/ analysing 
        trainVal, test = uf.split_trainVal_test_set(all_data = all_data)

    if test_all:
    # --- for instances where i want to perfrome test on the full dataset
    # --- could give infalated scores
        test = all_data

    # ---- loading a pretrained model ----
    #my_metrics = [keras.metrics.MeanAbsoluteError(),keras.metrics.MeanSquaredError(),NN_u.R2]
    my_met = NN_u.R2
    #,custom_objects = { 'my_loss_MSE_and_PE': my_loss}

    trained_mdl = keras.models.load_model(model_path,custom_objects = { 'R2': my_met})

    # ---- printing metrics ----
    X_test = test.loc[:,modle_inputs].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()
    
    
    y_pred = trained_mdl.predict(X_test)
    print(y_pred)
    if type == 'log':
            v1 = test.loc[:,'v1'].to_numpy().reshape(-1,1)
            y_pred = transf.inv_transf_Thung(y_pred,v1)
            print(y_pred.shape)
    if type == 'conductivity_test':
            
            test,_ = transf.transf_Thung(test,yname=yname)
            X_test = test.loc[:,modle_inputs].to_numpy()
            y_true = test.loc[:,['y_tilde']].to_numpy()
            y_pred = trained_mdl.predict(X_test)
    
    print(y_pred.shape)
    print('NN own calc R2 score:'+str(skm.r2_score(y_true,y_pred)))
    # loss,MAE,MSE,R2 = trained_mdl.evaluate(X_test,y_true,batch_size = 100, verbose = 2)  # returns loss and metrics
    # print("loss aka MSE : %.8f" % MSE)
    # print("MAE: %.8f" % MAE)
    # print("R2: %.8f" % R2)

    #----- eval hyperparms ----
    PE_trsh = 0.1
    error_trsh = 0.025
    if type == 'conductivity_test': error_trsh = 5

    uf.print_all_metrics(y_true,y_pred,full_name=model_path,
                        error_trsh=error_trsh,PE_threshold=PE_trsh)

    return
def analyse_ensembled(backend_type = 'simple',
    model_path_log = './NN_models/NN_gaus_log_03',
    model_path_lin = './NN_models/NN_gaus_07',
    model_path_backend = './NN_models/NN_backend_02',
    test_all = False,
    use_other_data = False,
    modle_inputs = ['v1','v2','v3','v4'],
    set_v4_null = False
):
# --- loading non standart data ---
    
    if use_other_data == 'ds3':
        yname = 'i_lat'
        scale_factor = -1E03
        all_data = pd.read_table("./data/set7_lat2_m13.tbl")
        all_data.columns =["v1","v2","v3",yname]
        all_data['v4'] = np.zeros(shape =(len(all_data['v3']),1))
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        test_all = True
    
    elif use_other_data == 'ds5':
        yname = 'i_lat'
        scale_factor = -1E03
        all_data = pd.read_csv("./data/Set5_v2.csv")
        all_data.columns =["v1","v2","v3","v4",yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        test_all = True
   
# --- loading the standart data Dataset ---
    else:
        yname = 'i_lat'
        scale_factor = 1E03
        all_data = pd.read_table("./data/id.tbl")
        all_data.columns =["v1","v2","v3","v4",yname]
        if set_v4_null:
                voltages =  {
                'v1' : None,
                'v2' : None,
                'v3' : None,
                'v4' : 0.0 }
                all_data = visres.select_data_one_const(voltages=voltages,df_full=all_data,epsilon=0.001)

        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)

        # --- using only unseen data for testing/ analysing 
        trainVal, test = uf.split_trainVal_test_set(all_data = all_data)

    if test_all:
    # --- for instances where i want to perfrome test on the full dataset
    # --- could give infalated scores
        test = all_data

    

    X_test = test.loc[:,modle_inputs].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()
  
    # ---- loading a pretrained model ----

    #my_metrics = [keras.metrics.MeanAbsoluteError(),keras.metrics.MeanSquaredError(),NN_u.R2]
    my_loss = NN_u.my_loss_Mp4E
    my_met = NN_u.R2
    #,custom_objects = { 'my_loss_MSE_and_PE': my_loss}

    mdl_lin = keras.models.load_model(model_path_lin,custom_objects = { 'R2': my_met})
    y_pred_lin = mdl_lin.predict(X_test)

    mdl_log= keras.models.load_model(model_path_log,custom_objects = { 'R2': my_met})
    sigma_log = mdl_log.predict(X_test)
    sigma_log = sigma_log.reshape(-1,1)
    v1 = test.loc[:,'v1'].to_numpy()
    v1 = v1.reshape(-1,1)
    y_log_pred = transf.inv_transf_Thung(sigma_log,v1)

   #---------actual msodel evalutaion--------
    if backend_type == 'simple':
        y_pred = NN_u.i_lin_log_backend(y_pred_lin=y_pred_lin,test_data=test,sigma_log=sigma_log)
    else :
        mdl_NN_backend = keras.models.load_model(model_path_backend,custom_objects = { 'R2': my_met})
        previous_preds = pd.DataFrame(
        {'y_lin_pred': y_pred_lin.reshape(-1), 
        'y_log_pred': y_log_pred.reshape(-1),
        'sigma_log': sigma_log.reshape(-1)}
        )
        X_test_backend = previous_preds.to_numpy()
        y_pred = mdl_NN_backend.predict(X_test_backend)
    print(y_pred.shape)
    print('NN own calc R2 score:'+str(skm.r2_score(y_true,y_pred)))
   
    #----- eval hyperparms ----
    PE_trsh = 0.1
    error_trsh = 0.025
    full_name = str(model_path_lin +'  with  '+ model_path_log)

   
    uf.print_all_metrics (y_true,y_pred,full_name=full_name,
                        error_trsh=error_trsh,PE_threshold=PE_trsh)
    return
def predict_with_single_NN_model(
    mdl_path:str = './NN_models/NN_15',
    mdl_type = 'lin',
    modle_inputs = ['v1','v2','v3','v4'],
    data: pd.DataFrame() = None ):

    X = data.loc[:,modle_inputs].to_numpy()
    my_met = NN_u.R2
    #,custom_objects = { 'my_loss_MSE_and_PE': my_loss}

    trained_mdl = keras.models.load_model(mdl_path,custom_objects = { 'R2': my_met})
    y_pred = trained_mdl.predict(X)
    if mdl_type == 'log':
        v1 = data.loc[:,['v1']]
        y_pred = transf.inv_transf_Thung(y_pred,v1)
    return y_pred

def predict_with_ensembled_NN_model(
    backend_type = 'simple',
    model_path_log = './NN_models/NN_gaus_log_03',
    model_path_lin = './NN_models/NN_gaus_07',
    model_path_backend = './NN_models/NN_backend_02',
    modle_inputs = ['v1','v2','v3','v4'],
    data: pd.DataFrame() = None ):
    
    

    X = data.loc[:,modle_inputs].to_numpy()
    my_met = NN_u.R2
    #,custom_objects = { 'my_loss_MSE_and_PE': my_loss}
    mdl_lin = keras.models.load_model(model_path_lin,custom_objects = { 'R2': my_met})
    y_pred_lin = mdl_lin.predict(X)

    mdl_log= keras.models.load_model(model_path_log,custom_objects = { 'R2': my_met})
    sigma_log = mdl_log.predict(X)
    sigma_log = sigma_log.reshape(-1,1)
    v1 = data.loc[:,'v1'].to_numpy()
    v1 = v1.reshape(-1,1)
    y_log_pred = transf.inv_transf_Thung(sigma_log,v1)

    if backend_type == 'simple':
        y_pred = NN_u.i_lin_log_backend(y_pred_lin=y_pred_lin,test_data=data,sigma_log=sigma_log)
    else :
        mdl_NN_backend = keras.models.load_model(model_path_backend,custom_objects = { 'R2': my_met})
        previous_preds = pd.DataFrame(
        {'y_lin_pred': y_pred_lin.reshape(-1), 
        'y_log_pred': y_log_pred.reshape(-1),
        'sigma_log': sigma_log.reshape(-1)}
        )
        X_test_backend = previous_preds.to_numpy()
        y_pred = mdl_NN_backend.predict(X_test_backend)
    return y_pred
def name_list_NN(name = 'NNs2'):
     if name == 'NNs2': i = 0
     elif name == 'NNs3': i = 1
     elif name == 'NNs6': i = 2
     elif name == 'NNs7': i = 3
     else:
        raise NameError('Model name unkown')
     name_list_lin = ['./NN_models/Set2_NNlin_04','./NN_models/Set3_NNlin_03','./NN_models/Set6_m13_NNlin_01','./NN_models/Set7_NNlin_01'
                      ,'Set2_lin_Tune1','Set3_lin_Tune1','Set6_lin_Tune3','Set7_lin_Tune1']
     name_list_log = ['./NN_models/Set2_NNlog_04','./NN_models/Set3_NNlog_05','./NN_models/Set6_NNlog_10','./NN_models/Set7_NNlog_03',
                      'Set2_transfer_Tune1','Set3_transfer_Tune1','Set6_transfer_Tune1','Set7_transfer_Tune1']
     return (name_list_lin[i],name_list_log[i])
if __name__ == "__main__":
    main()
