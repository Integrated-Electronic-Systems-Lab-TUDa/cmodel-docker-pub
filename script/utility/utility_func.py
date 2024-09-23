import pandas as pd
import numpy as np
import sklearn.metrics as skm 
from matplotlib import pyplot as plt
import numpy.ma as ma
from sklearn.model_selection import train_test_split
from itertools import product
import tensorflow as tf
import keras.backend as K
import kennard_stone as ks

# --- utility functions used in different parts throughout the other methods ---
def extract_all_data (tbl_path_original = './data/id.tbl',yname = 'y',
    scale = False, factor = 1E03, columns = 5,ftype = 'tbl'
):
    """
    Extracts the original simulation data from the tbl.files
    Args:
        tbl_path_original: Path specifing the .tbl file with the original data
        yname: how y var shall be named 
        scale: Weather or not to scale y 
        factor: scale factor to use when scaling. y = y * factor

    Returns: 
        df_data: df containing the data with colums: ["v1","v2","v3","v4",yname]
    """
    if tbl_path_original == None:
        raise ValueError('path variable should not be None, needs to be set')
    if ftype == 'csv':
        df_data = pd.read_csv(tbl_path_original)
    else:
        df_data = pd.read_table(tbl_path_original)

    if columns == 4:
        df_data.columns =["v1","v2","v3",yname]
    else:    
        df_data.columns =["v1","v2","v3","v4",yname]
    if scale:
        df_data[yname] = df_data[yname].apply(lambda x: x*factor)

    return df_data

def critical_values (y_true, y_pred, PE_trsh = 0.3,error_trsh = 0.05):
    """
    Returns:
        df_Y_Errors: df with all Error values (y_ture,y_pred,PE,AE,logAcc) for all input pairs (y_ture,y_pred)
        df_Y_crit: df with all the crictical y_ture, y_pred pairs.
        df_high_AE: df with all y_ture,y_pred pair where the AE is above the 'error_trsh'
        
        Paris are condiderd critical if the percentage Error is larger than 'PE_trsh' 
        and the Abolute Error is larger than 'error_trsh'. This is necessarry to deal with pairs where y_ture is e.g. 0.001E-07 (essentialy 0.0) 
        but y_pred is 0.001E-06 (also essentially 0.0, but an percentage whise error of 900% !)   
    """
    # first calculate the absolte error; only if absolute error is larger than medians abolute error perced
    # setting lowe boundary to 1pA 
    epsilon = np.ones(shape =(y_true.size,1))*1E-9

    median_absolute_error = skm.median_absolute_error(y_true=y_true, y_pred = y_pred)
    AE = np.zeros (shape = np.shape(y_pred))
    np.absolute(y_pred - y_true,out=AE)
    percentage_error = np.zeros (shape = np.shape(y_pred))
    np.absolute(np.true_divide(AE,np.maximum(epsilon,np.absolute(y_true))),out = percentage_error)  
    logAcc = log_Acc(y_true,y_pred)

    Y_Errors = np.concatenate((y_true,y_pred,percentage_error,AE,logAcc),axis=1)
  
    df_Y_Errors = pd.DataFrame(Y_Errors,columns=['y_ture_val','y_pred_val','PE','AE','logAcc'])

    # --- extract critical values for data frame with all Error Values  
    df_Y_crit = df_Y_Errors[(df_Y_Errors['PE'] > PE_trsh) & (df_Y_Errors['AE'] > error_trsh)] 
    # --- extract prediction ture pairs with high absolute Error
    y_high_AE = df_Y_Errors.loc[lambda df: df['AE'] > error_trsh,:]
 
    df_Y_crit = df_Y_crit.sort_values(by='AE',ascending = False)

    return df_Y_Errors,df_Y_crit,y_high_AE

def print_all_metrics(y_true, y_pred, full_name = 'qlattice',
                        error_trsh = None, PE_threshold = 0.5):

    """ Prints all metrices used for evaluating prediction performance
        Very important method. Used to ensure equal evalution for all kinds of model types.
    """

    Y_errors_left,y_crit_left,y_high_AE_left= critical_values(y_true=y_true,y_pred=y_pred,error_trsh=error_trsh,PE_trsh=PE_threshold)
    #---------actual model evalutaion--------
    print(f'')
    print('---- Analysis of: '+full_name + ' ---' )
    print(f'R2 current model: %.6f' % skm.r2_score(y_true,y_pred))
    print('MAE current model: %.4E' %(skm.mean_absolute_error(y_true,y_pred)))
    print('MSE current model: %.4E' %(skm.mean_squared_error(y_true,y_pred)))
    print('MAPE current model: %.4E' %(skm.mean_absolute_percentage_error(y_true,y_pred)))
    print('sMAPE current model: %.6f' %(Smape(y_true,y_pred)))
    print('Tweedie Dev current model: %.4E' %(tweedie_dev(y_true,y_pred)))

    print('--- All Errors ---')
    print(Y_errors_left.describe(percentiles=[0.5,0.9]))
   
    

    print('--- Critical; than is: AE > '+str(error_trsh)+' and PE > '+ str(PE_threshold)+'---')
    print(y_crit_left.describe(percentiles=[0.5,0.9]))

    print('--- AE greater than: '+str(error_trsh)+' ---')
    print(y_high_AE_left.describe(percentiles=[0.5,0.9]))
    Y_errors_left.sort_values(by=['PE'], ascending= False,inplace=True)
    print(Y_errors_left.head(5))
    Y_errors_left.sort_values(by=['AE'], ascending= False,inplace=True)
    print(Y_errors_left.head(5))
  
    return  

def plot_correlation(df_to_plot):
    """ Creating correlation plots for each of the columns combinations of the given dataframe. (e.g. how correlated are col 1 and col 2) 
    """
    n_vars = df_to_plot.shape[1]

    S = 0 
    for x in range (0,n_vars):
            S = S+x
    if S % 2 == 1 :
        S = S+1
    n_rows = int(S/2) 
    pair_list = list()
    for a in range (0,n_vars):
        for b in range (a+1, n_vars):
            pair_list.append((a,b))

    fig, axs = plt.subplots(n_rows,2,layout="constrained")
    idx = 0
    for row in range(0,n_rows):
        for col in range (0,2):
                    
                    d_r = pair_list[idx][0]
                    d_c = pair_list[idx][1]
                    axs[row,col].plot(df_to_plot.iloc[:,d_r],df_to_plot.iloc[:,d_c],'b+')
                    axs[row,col].set_xlabel(df_to_plot.columns[d_r])
                    axs[row,col].set_ylabel(df_to_plot.columns[d_c])
                    idx = idx+1
                
            
    plt.show()
    return


# class loss_sMAPE(tf.keras.losses.Loss):
#     def call(self, y_true ,y_pred):
#         """ Calculates the sMAPE metric, for given yture and ypred pairs. 
#         sMAPE is an acronym for symmetric mean absolute percantage Error. This metric was developed to fight the flaws of the standart MAPE.
#         On of the mst imprortant properties, is the fact the maximum sMAPE for any prediction - ture pair is 200%. More on Wikipedia. 
#         """


#         # if(len(y_true) != len(y_pred)):
#         #     raise ValueError(f"y_ture and y_pred must be of equal length {len(y_true)} and {len(y_pred)}")
#         n = len(y_true)

#         AE = K.abs(K.subtract(y_true,y_pred))
#         # AE = np.abs(y_true-y_pred)
#         denom = K.mult(K.add(K.abs(y_true),K.abs(y_pred)) , 0.5)
        
#         a = np.zeros(shape = (len(y_true),1))
#         for i in range(len(y_true)):
#             if not ((AE[i,0] == 0.0) & (denom[i,0] == 0.0)):
#                 a[i,:] =  np.divide(AE[i,0] ,denom[i,0] )
#         sMAPE = np.mean(a)
#         return sMAPE
    
# class decreasingMSE(tf.keras.losses.Loss):
#     def call(self, y_true ,y_pred):
 
#         my_loss = tf.keras.losses.MeanSquaredError()

#         decMSE = my_loss.call(y_true, y_pred)

#         return K.decMSE * 

def Smape(y_true : np.ndarray ,y_pred: np.ndarray):
    """ Calculates the sMAPE metric, for given yture and ypred pairs. 
        sMAPE is an acronym for symmetric mean absolute percantage Error. This metric was developed to fight the flaws of the standart MAPE.
        On of the mst imprortant properties, is the fact the maximum sMAPE for any prediction - ture pair is 200%. More on Wikipedia. 
    """
    if (y_true.shape[0] < y_true.shape[1]):
        raise ValueError('must be row arrays')
    
    if(y_true.shape != y_pred.shape):
        raise ValueError('y_ture and y_pred must be of equal length')
    n = len(y_true)
    AE = np.abs(y_true-y_pred)
    denom = ((np.abs(y_true)+np.abs(y_pred)) * 0.5)
    a = np.zeros(shape = (len(y_true),1))
    for i in range(len(y_true)):
        if not ((AE[i,0] == 0.0) & (denom[i,0] == 0.0)):
            a[i,:] =  np.divide(AE[i,0] ,denom[i,0] )
    sMAPE = np.mean(a)
    return sMAPE


def sAPE(y_true : np.ndarray ,y_pred: np.ndarray):
    """ Calculates the sAPE metric, for given yture and ypred pairs. 
        sMAPE is an acronym for symmetric mean absolute percantage Error. This metric was developed to fight the flaws of the standart MAPE.
        On of the mst imprortant properties, is the fact the maximum sMAPE for any prediction - ture pair is 200%. More on Wikipedia. 
    """
    if(len(y_true) != len(y_pred)):
        raise ValueError('y_ture and y_pred must be of equal length')
    n = len(y_true)
    AE = np.abs(y_true-y_pred)
    denom = ((np.abs(y_true)+np.abs(y_pred)) * 0.5)
    a = np.zeros(shape = (len(y_true),1))
    for i in range(len(y_true)):
        if not ((AE[i,0] == 0.0) & (denom[i,0] == 0.0)):
            a[i,:] =  np.divide(AE[i,0] ,denom[i,0] )
    # sMAPE = np.mean(a)
    return a

def tweedie_dev(y_true, y_pred):
    """ Used because of its simillarity to log(acc)
        cant take zero or negativ input values 
    """
    #--- creating a mask whih will exclued all zeros
    y_true_abs = np.abs(y_true)
    y_pred_abs = np.abs(y_pred)
    
    y_true_abs_valid,y_pred_abs_valid = make_greater_zero(y_true_abs,y_pred_abs)
    
    twd_dev = skm.mean_tweedie_deviance(y_true_abs_valid,y_pred_abs_valid,power=2)
    return twd_dev

def log_Acc(y_true,y_pred):
    """ Computes the log_acc for every valid true,pred pair 
    """
    y_true_abs = np.abs(y_true)
    y_pred_abs = np.abs(y_pred)
    #---  marks all data as invalid which is equal to zero
    mask_ytrue = ma.masked_equal(y_true_abs, value=0.0).mask

    #--- remove all ture & pred pairs where ture is equal to zero
    y_true_abs_valid = np.ma.array(y_true, mask = mask_ytrue)
    y_pred_abs_valid = np.ma.array(y_pred,mask=mask_ytrue)

    # --- worrks because numpy masked arrays will return '--' when function is not defined for the given inputs
    log_acc_masked = np.abs(ma.log(ma.divide(y_pred_abs_valid,y_true_abs_valid)))
    # --- to make in normal nd.array again:
    log_acc = log_acc_masked.filled(fill_value = np.nan)
    return log_acc

def make_greater_zero(y_true,y_pred):
    """ Cuts y_ture y_pred pairs where one of the two is equal to zero.
    """
     #---  marks all data as invalid which is equal to zero
    mask_ytrue = ma.masked_equal(y_true, value=0.0).mask
    mask_ypred = ma.masked_equal(y_pred, value=0.0).mask
    #--- mask which is true (invalid data) if one of the base masks are true
    mask_both = ma.mask_or(mask_ytrue,mask_ypred)

    y_true_abs_valid = np.ma.array(y_true, mask = mask_both).compressed()
    y_pred_abs_valid = np.ma.array(y_pred,mask=mask_both).compressed()

    return y_true_abs_valid,y_pred_abs_valid

def bound_current(all_data: pd.DataFrame, ilat_col: str,lower_bound = 1E-15) -> pd.DataFrame:
    """ Bounds all currents to a fixed lower absolute (e.g, 1E-15). All currents which are smaller than this 
        value are set to it.
        Args:
            all_data: dataframe with a colum which shall be processed
            ilat_col: colum name of the colum to process
            lower_bound: absolute bound value
        Returns: 
            pd.DataFrame with the current set to the given value
    """
    lower_bound = np.abs(lower_bound)
    all_data.loc[(all_data[ilat_col] >= 0) & (all_data[ilat_col] <= lower_bound), ilat_col] = lower_bound
    all_data.loc[(all_data[ilat_col] <= 0) & (all_data[ilat_col] >= (-1)*lower_bound), ilat_col] = (-1) *lower_bound

    return all_data

def split_trainVal_test_set(all_data = pd.DataFrame(),yname = 'y', test_size = 0.15):
    """ splits a well defined test set for final model evaluation.
        Also gives us a validation and train set.
        vald_set: -> for hyperparameter tuning and cross model eval
        test_set: -> for final model eval. would be even better to have a complete new set here
                     In our case we can also comapre against new TCAD calcs. 
        train_set -> for training the model parameters
    """
    if all_data.empty:
        all_data = extract_all_data(tbl_path_original = './data/id.tbl',yname=yname,scale=True)

    #--- random state for generating test set is fixed so that always the same 15% are extracted as test data
    #--- and therefore can never be part of the train set 
    trainVal,test = train_test_split(all_data,test_size=test_size,random_state=9,shuffle=True)
    
    return trainVal,test

def generate_param_product_list(param_grid: dict):
    """ Takes an dict with a Hyperparm-grid as input and transfers it 
        to a list of dicts, where each dict is one unique Hyperparms combination.
        Args:
            param_grid: Dict containing all the hyperparams and their respective value sets.
        Return:
            p_product_list: List containing dicts. Each dict features on unique Hyperparm cobination.
        Example:
            param_gird = {'A': [1, 2, 3], 'B': [5, 6, 7]}
            out = [{'A': 1, 'B': 5}, {'A': 1, 'B': 6}, {'A': 1, 'B': 7}, {'A': 2, 'B': 5}, 
                    {'A': 2, 'B': 6}, {'A': 2, 'B': 7}, {'A': 3, 'B': 5}, {'A': 3, 'B': 6}, 
                    {'A': 3, 'B': 7}]
    """
    p = param_grid

    items = sorted(p.items())
    #--- returns dict as list of tuples
    #--- * is the unpacking operator
    keys, values = zip(*items)
    print('--- keys received: ---')
    print(keys)
    print('--- values received: ---')
    print(values)
  
    p_product_list = [dict(zip(keys, v)) for v in product(*values)]
    print('--- first 3 hyperparam_dicts: ---')
    print(p_product_list[0:3])
    return p_product_list
def expand_TCAD_Set4(df_set4: pd.DataFrame, VP = 0.0):
    """ Expands the Dataset generated by the TCAD sweeps in such a way, that the structure is like the one from Set1-3
        The Set4 must have the structure(columns) [X(Vfg=-1.6),Y(Vfg=-1.6),X(Vfg=0.8), .... Y(Vfg=1.6)] the columns can be of different length
    """
    Vfg_values = [-1.6,-0.8,0.0,0.8,1.6]
    if len(df_set4.columns) != len(Vfg_values)*2:
         print('Set4 Data has an unexpected number of columns; cant continue')
         return
    df_lengths = df_set4.count(axis=0) #how many non nan arguments are there in each column
    df_out = pd.DataFrame()
    for k, vfg in enumerate (Vfg_values):
        df_temp = df_set4.iloc[:,2*k:2*k+2].dropna()
        df_temp.reset_index()
        vfg_col = np.ones(len(df_temp.index))*vfg
        df_temp.columns =['v1','ilat2_ds4']
        df_temp['v1'] =df_temp['v1'].apply(lambda x: x-1.6)
        df_temp['ilat2_ds4'] =df_temp['ilat2_ds4'].apply(lambda x: x*1)
        df_temp.insert(1,'v2',vfg_col)
        #print(df_temp.head(10))
        df_out = pd.concat([df_out,df_temp],axis = 0,ignore_index=True)
    vp = np.ones(len(df_out['v2']))*VP
    df_out.insert(2,'v3',vp)
    df_out.insert(3,'v4',vp)
    print('--- Expanded Dataset 4 ---')
    print(df_out.info())
    print(df_out.describe())
    print(df_lengths.head(10))
    return df_out
def save_TCAD_Set2_as_Dataset4():
    df_all = pd.DataFrame()
    df_1 = read_in_TCAD_S2_csv('./plot_data/Set5_v4_n1_6.csv')
    df_1.columns= ['v1','v2','v3','v4','ilat2_ds4']
   # df_1 = expand_TCAD_Set4(df_1,VP = -1.6)
    df_all = pd.concat([df_all,df_1],axis = 0,ignore_index=True)
    df_2 = read_in_TCAD_S2_csv('./plot_data/Set5_v3_for_VP_n0_8.csv')
    df_2 = expand_TCAD_Set4(df_2,VP = -0.8)
    df_all = pd.concat([df_all,df_2],axis = 0,ignore_index=True)
    df_3= read_in_TCAD_S2_csv('./plot_data/Set5_v2_for_VP_p0_0.csv')
    df_3 = expand_TCAD_Set4(df_3,VP = 0.0)
    df_all = pd.concat([df_all,df_3],axis = 0,ignore_index=True)
    df_4= read_in_TCAD_S2_csv('./plot_data/Set5_v2_for_VP_p0_8.csv')
    df_4 = expand_TCAD_Set4(df_4,VP = 0.8)
    df_all = pd.concat([df_all,df_4],axis = 0,ignore_index=True)
    df_5 = read_in_TCAD_S2_csv('./plot_data/Set5_v2_for_VP_p1_6.csv')
    df_5 = expand_TCAD_Set4(df_5,VP = 1.6)
    df_all = pd.concat([df_all,df_5],axis = 0,ignore_index=True)
    print(df_all.describe())
    print(df_all.info())
    df_all.to_csv(path_or_buf = './data/Set5_v4.csv',index=False,header = False)
    return
def read_in_TCAD_S2_csv(path = './plot_data/S2_Ilat2_over_Vlat2_for_VP_n1_6V.csv'):
    
    df_data = pd.read_csv(path,na_values='-')
    #print(list(df_data.columns))
    #print(df_data.describe())
    print(df_data.head(5))
    print(df_data.tail(5))
    return df_data





def evaluate_test_lin(model, test, Xlabels=["Vlat21", "Vfglat1", "Vtglat1"], Ylabel=["ids"]):
    Xlabels =   ["Vlat21", "Vfglat1", "Vtglat1"]

    R2 = skm.r2_score(test.loc[:,Ylabel],model.predict(test[Xlabels]))
    sMAPE = Smape(test.loc[:,Ylabel].to_numpy(),model.predict(test[Xlabels]))

    return R2, sMAPE



def evaluate_test_log(model, test, Xlabels=["Vlat21", "Vfglat1", "Vtglat1"], Ylabel=["ids"]):
    Xlabels =   ["Vlat21", "Vfglat1", "Vtglat1"]


    test_backtransform = pd.DataFrame()

    y_pred = model.predict(test[Xlabels])
    test_backtransform["ids"] = -pd.DataFrame((np.e**(y_pred))).multiply(test["Vlat21"].values.reshape(-1,1)) 
    
    #test_backtransform["ids"] = pd.DataFrame(-np.multiply( model.predict(test[Xlabels]),test["Vlat21"].values.reshape(-1,1)))
    
    R2 = skm.r2_score(test.loc[:,Ylabel],test_backtransform["ids"])
    sMAPE = Smape(test.loc[:,Ylabel].to_numpy(),test_backtransform["ids"].to_numpy().reshape(-1,1))

    return R2, sMAPE

def evaluate_test_log_all(model, test, Xlabels=["Vlat21", "Vfglat1", "Vtglat1"], Ylabel=["ids"]):
    Xlabels =   ["Vlat21", "Vfglat1", "Vtglat1"]


    test_backtransform = pd.DataFrame()

    y_pred = model.predict(test[Xlabels])
    test_backtransform["ids"] = -pd.DataFrame((np.e**(np.array(y_pred)))).multiply(test["Vlat21"].values.reshape(-1,1)) 
    
    #test_backtransform["ids"] = pd.DataFrame(-np.multiply( model.predict(test[Xlabels]),test["Vlat21"].values.reshape(-1,1)))
    
    R2 = skm.r2_score(test.loc[:,Ylabel],test_backtransform["ids"])
    mae = skm.mean_absolute_error(test.loc[:,Ylabel],test_backtransform["ids"])
    mse = skm.mean_squared_error(test.loc[:,Ylabel],test_backtransform["ids"])
    sMAPE = Smape(test.loc[:,Ylabel].to_numpy(),test_backtransform["ids"].to_numpy().reshape(-1,1))

    return R2, mae, mse, sMAPE