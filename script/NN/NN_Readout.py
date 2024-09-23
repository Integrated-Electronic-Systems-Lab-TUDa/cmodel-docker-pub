import sys
sys.path.insert(1,'c:\\Users\\jowil\\OneDrive\\Dokumente\\Uni\\MSc_4_Semester\\020_python\\021_symbolic_reg')

import numpy as np
import pandas as pd
from tensorflow import keras 
from keras import layers
import NN.NN_utility as NN_u
import h5py
import utility_func as uf
from sklearn.model_selection import train_test_split

#--- methods for extracting weights and biases from a pretrained NN Model, saving Weights and Biases, and testing calculation with the extracted weights and biases

def main():
    print('---start---')
    #build_test_NN()
    #train_test_NN()
    w = save_weights(save_path = './NN_models/mdl_weights/Set3_NNlog_05.hdf5',model_path ='./NN_models/Set3_logTune3')
    #check_if_saved_and_loaded_weights_are_equal(weights_saved=w,weights_path='./NN_models/mdl_weights/NN_V_logTune3.hdf5')
    #load_weights(mypath = './NN_models/mdl_weights/NN_V_log_06.hdf5' )
    #pred_with_weights()
    #test_mdl(model_path='./NN_models/NN_logTune3',weights_path='./NN_models/mdl_weights/NN_V_logTune3.hdf5')
    #save_weights_to_csv()
    return

def build_test_NN(
    n_neurons =32, dim_in = 4
):
    # --- building input layer----
    x = layers.Input(shape= (dim_in),name ="inputlayer")

    # --- hidden layser ----
    zOne = layers.Dense(units=n_neurons, activation = "tanh", name = "hiddenlayer1")(x)
    zTwo = layers.Dense(units=n_neurons, activation = "tanh", name = "hiddenlayer2")(zOne)
    zThree = layers.Dense(units=n_neurons, activation = "tanh", name = "hiddenlayer3")(zTwo)

    # --- output Layer
    y = layers.Dense(units = 1, activation = 'linear', name = "outputlayer")(zThree)

    # --- build model
    mdl = keras.Model(inputs=x,outputs=y, name = "test_nn")
    print(mdl.summary())

    # --- Setting metrics and compiling
    #my_metrics = [keras.metrics.MeanAbsoluteError(),NN_u.R2]
    my_loss = keras.losses.MeanSquaredError()
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=2E-03),loss=my_loss)

    return mdl

def train_test_NN(mdl_save_path = './NN_models/TestModel01'):
    
    # --- data loading and splitting
    yname = 'id'
    scale_factor = 1E03
    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4",yname]
    all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
    trainVal,test = uf.split_trainVal_test_set(all_data=all_data)

    # --- creating the validation dataset for this NN train run, can be different random seed every time
    train,val_data = train_test_split(trainVal,test_size=0.1,random_state=20)

    compiled_mdl = build_test_NN()

    # ---- running fit -----
    X = train.loc[:,['v1','v2','v3','v4']].to_numpy()
    y = train.loc[:,[yname]].to_numpy()
    #--- validation data is currently not used since we are not tuning any hyperparameter yet
    X_val = val_data.loc[:,['v1','v2','v3','v4']].to_numpy()
    y_val = val_data.loc[:,[yname]].to_numpy()

    print('----- fitting the model-----')
    history = compiled_mdl.fit(X,y,batch_size = 100,epochs = 50,
    validation_data=(X_val,y_val),verbose = 2)

    compiled_mdl.save(mdl_save_path)

    
    # ---- printing metrics ----
    X_test = test.loc[:,['v1','v2','v3','v4']].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()


    y_pred = compiled_mdl.predict(X_test,batch_size = 100)
    print('np calc R2 score on test:'+str(NN_u.R2_np(y_true,y_pred)))

    return

def save_weights(save_path = './NN_models/mdl_weights/test.hdf5',model_path ='./NN_models/TestModel01' ):
    my_met = NN_u.R2
    #,custom_objects = { 'my_loss_MSE_and_PE': my_loss}

    my_model= keras.models.load_model(model_path,custom_objects = { 'R2': my_met})
    weights = my_model.get_weights()
    my_model.save_weights(filepath = save_path ,save_format = 'h5')
    return weights

def load_weights(mypath = './NN_models/mdl_weights/test.hdf5'):

    filename = mypath
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        all_group_names = f.keys()
        print("Keys: %s" % all_group_names)

        Layer_list = list()
        name_list = list()
        weight_list = list()

        for top_grp_name in all_group_names:
           # print ('Name: %s ' %top_grp_name)
           # print(type(f.get(top_grp_name)))
           # --- get all the subgroups of the topgrps e.x: top: HiddenLayer1/ sub: HiddenLayer1
            for subgrp_name in f.get(top_grp_name).keys():
                Layer_list.append(subgrp_name)
            # --- get all the datasets for each Layer (Weights/Kernal and Bias)
                for weight_arr_items in f.get(top_grp_name).get(subgrp_name).items():
                    name_list.append(weight_arr_items[0])
                    # returns the dataset as np.array
                    weight_list.append(weight_arr_items[1][()])

        #print(Layer_list)
        #print(name_list)
        #print(weight_list)

    return Layer_list,name_list,weight_list

def test_mdl(
    model_path = './NN_models/TestModel01',
    test_all = True,
    mdl_type = 'self_imp',
    weights_path = './NN_models/mdl_weights/test.hdf5'
):
    # --- loading train data -----
    yname = 'ilat'
    scale_factor = 1E03

    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4",yname]
    all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)

    trainVal, test = uf.split_trainVal_test_set(all_data = all_data)

    if test_all:
    # --- for instances where i want to perfrome test on the full dataset
    # --- could give infalated scores
        test = all_data

    PE_trsh = 0.1
    error_trsh = 0.025

    my_met = NN_u.R2
    #,custom_objects = { 'my_loss_MSE_and_PE': my_loss}

    trained_mdl= keras.models.load_model(model_path,custom_objects = { 'R2': my_met})

    X_test = test.loc[:,['v1','v2','v3','v4']].to_numpy()
    y_true = test.loc[:,[yname]].to_numpy()
    
    

    if mdl_type == 'self_imp':
        print('weights path used for prediction %s' %weights_path)
        y_pred_own = pred_with_weights_flexibleLayers(X_test,weights_path=weights_path).reshape(-1,1)
        print('manual np.matmul pred:')
        print(y_pred_own[10:15,:])

        #y_pred_own_calc = pred_As_in_verilog_A_vec(X_test).reshape(-1,1)
        #print('manual clac pred:')
        #print(y_pred_own_calc[10:15,:])
       
    
        y_pred = trained_mdl.predict(X_test)
        print('actual tf NN pred: ')
        print(y_pred[10:15,:])
        print('true value : ')
        print(y_true[10:15,:])

        print('tf NN pred - np.matmul pred:')
        df_controll = pd.DataFrame(np.abs(y_pred_own-y_pred))
        print(df_controll.describe())

        #print('np.matmul pred - own clac pred:')
        #df_controll2 = pd.DataFrame(np.abs(y_pred_own-y_pred_own_calc))
        #print(df_controll2.describe())

        

    uf.print_all_metrics (y_true,y_pred_own,full_name=model_path,
                        error_trsh=error_trsh,PE_threshold=PE_trsh)
    return

def pred_with_weights_matmul(X_test):
    Layer_list,name_list,weight_list = load_weights()
    x = X_test

    W0 = weight_list[1]
    b0 = weight_list[0]
    W1 = weight_list[3]
    b1 = weight_list[2]
    W2 = weight_list[5]
    b2 = weight_list[4]

    zl1 = np.matmul(x,W0) + b0.reshape(1,-1)
    zl1 = np.tanh(zl1)
   
    zl2 = np.matmul(zl1,W1) + b1.reshape(1,-1)
    zl2 = np.tanh(zl2)

    out = np.matmul(zl2,W2) + b2.reshape(1,-1)

    return out

def save_weights_to_csv(mypath = './NN_models/mdl_weights/test.hdf5'):
    Layer_list,name_list,weightbias_list = load_weights(mypath=mypath)
    save_csv_path = './NN_models/mdl_weights/test01_WB_as_csv'

    for n in range(len(Layer_list)):
        print('calc Layer: %s'%Layer_list[n])
        path_w = save_csv_path +'_'+ str(Layer_list[n]) + '_w.csv'
        path_b = save_csv_path +'_'+ str(Layer_list[n]) + '_b.csv'
       
        curr_weight = weightbias_list[2*n+1]
        np.savetxt(path_w, X=curr_weight, delimiter=",")
        curr_bias = weightbias_list[2*n]
        np.savetxt(path_b, X=curr_bias, delimiter=",")
    return

def pred_with_weights_ownmul(X_test):
    Layer_list,name_list,weight_list = load_weights()
    

    W0 = weight_list[1]
    b0 = weight_list[0]
    W1 = weight_list[3]
    b1 = weight_list[2]
    W2 = weight_list[5]
    b2 = weight_list[4]
        

    out = np.zeros(shape=(X_test.shape[0],1))

    for nth_row in range(X_test.shape[0]):
        x_row = X_test[nth_row,:]
        x = x_row.reshape(-1,1)

        # --- frist layer 
        zl1 = mat_times_vec(W0.transpose(),x)+ b0.transpose().reshape(-1,1)
        zl1 = np.tanh(zl1)

        # --- second layer
        zl2 = mat_times_vec(W1.transpose(),zl1)+ b1.transpose().reshape(-1,1)
        zl2 = np.tanh(zl2)

        # --- gives sigluar value (current prediction for the given input combination)
        # --- output layer
        out[nth_row,0] = mat_times_vec(W2.transpose(),zl2) + b2.transpose().reshape(-1,1)

    return out

def pred_with_weights_flexibleLayers(X_test,weights_path = './NN_models/mdl_weights/test.hdf5'):
    Layer_list,name_list,weightbias_list = load_weights(mypath=weights_path)
    x = X_test

    for n in range(len(Layer_list)):
        print('calc Layer: %s'%Layer_list[n])
        curr_weight = weightbias_list[2*n+1]
        curr_bias = weightbias_list[2*n]
        
        # --- input layer
        if n == 0:
            zl = np.matmul(x,curr_weight) + curr_bias.reshape(1,-1)
            zl = np.tanh(zl)
            print('Frist Weight: %s'%name_list[2*n+1])
            print(curr_weight.shape)
            print(curr_weight)
        # --- output Layer
        elif n == (len(Layer_list)-1):
            out = np.matmul(zl,curr_weight) + curr_bias.reshape(1,-1)
        # --- any hidden Layer
        else:
            zl = np.matmul(zl,curr_weight) + curr_bias.reshape(1,-1)
            zl = np.tanh(zl)

    return out

def pred_As_in_verilog_A_vec(X_test):
    Layer_list,name_list,weightbias_list = load_weights()
    # --- onyl works for Testmodel01 !
    # --- the weights and biases will defined as constant arrays at the begin of the Verilog-A file
    # --- might be easier to got for form where everything is row based (vec x M ) instead of (M x vec)
    W0 = weightbias_list[1].transpose()
    b0 = weightbias_list[0].reshape(-1,1)
    W1 = weightbias_list[3].transpose()
    b1 = weightbias_list[2].reshape(-1,1)
    W2 = weightbias_list[5].transpose()
    b2 = weightbias_list[4].reshape(-1,1)
        

    out = np.zeros(shape=(X_test.shape[0],1))

    for nth_row in range(X_test.shape[0]):

        x_row = X_test[nth_row,:]
        x = x_row.reshape(-1,1)

        # --- frist layer 
        zl1 = mat_times_vec(W0,x)+ b0
        zl1 = np.tanh(zl1)

        # --- second layer
        zl2 = mat_times_vec(W1,zl1)+ b1
        zl2 = np.tanh(zl2)

        # --- gives sigluar value (current prediction for the given input combination)
        # --- output layer
        out[nth_row,0] = mat_times_vec(W2,zl2) + b2

    return out

def mat_times_vec(M: np.ndarray, vec = np.ndarray):
    """ Implements a simple matrix multiplication of out = M x v.
        Args:
            M: np.ndarray of shape n x m
            vec: np.ndarray of shape v x 1
        Returns:
            out: np.ndarray of shape n x 1. 
    
    """
    
    if not (M.shape[1]== vec.shape[0]):
        raise TypeError('M and vec shapes do not match for Multiplictaion')
    if not (vec.shape[1]== 1):
        raise TypeError('vec does not have a single colum ')
    
    z = np.zeros(shape= (M.shape[0],1))
    for j in range(M.shape[0]):
        # --- taking a row of the matrix
        M_row = M[j,:]
        # --- multiplication of one row times the vector 
        for k in range(vec.shape[0]):
            z[j] = M_row[k] * vec[k,0] + z[j] 
    out = z
    return out

def vec_times_mat(M: np.ndarray, vec = np.ndarray):
    """ Implements a simple matrix multiplication of out = v x M.
        Args:
            M: np.ndarray of shape n x m
            vec: np.ndarray of shape 1 x n
        Returns:
            out: np.ndarray of shape 1 x m. 
    
    """
    
    if not (M.shape[0]== vec.size):
        raise TypeError('M and vec shapes do not match for Multiplictaion')
    if not (vec.shape[0]== 1):
        raise TypeError('vec does not have a single row ')
    
    z = np.zeros(shape= (1,M.shape[1]))
    for j in range(M.shape[1]):
        # --- taking a col of the matrix
        M_col = M[:,j]
        # --- multiplication of one col times the vector 
        for k in range(vec.size):
            z[0,j] = M_col[k] * vec[0,k] + z[0,j] 
    out = z
    return out

def check_if_saved_and_loaded_weights_are_equal(weights_saved, weights_path = 'NN_V_log_06.hdf5'):
    Layer_list,name_list,weightbias_list = load_weights(mypath=weights_path)
    print('--- weights from keras: ----')
    print(weights_saved[0:2])
    print('--- weights from custom list: ----')
    print(weightbias_list[0:2])
    print('--- checking diff for w0 and bo ---')
    diff_df = pd.DataFrame(weights_saved[1] - weightbias_list[0])
    print(diff_df.describe())
    
    return

if __name__ == '__main__':
    main()