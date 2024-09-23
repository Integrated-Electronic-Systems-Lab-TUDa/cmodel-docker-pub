import pandas as pd
import numpy as np 


def const_transf(data: pd.DataFrame = None):
    """
    Applies one of the implemented transfer frunctions to the given Dataset
    Returns the datafarme expanded by a colum which contains the new calculated y_tilde 
    eg. y_tilde = log_10(id) would be a reasonable transfer function
    """
    data = data[(data['v1'] != 0)]
    y = data.loc[:,'id'].to_numpy()
    yy = (y + (abs(np.min(y))*(1+1E-10)))  
    y_tilde = np.log(yy)

    data_aug = data.copy()
    data_aug['id'] = y_tilde
    data_aug.columns = ['v1','v2','v3','v4','y_tilde']

    return data_aug,data

def inv_const_trans(y_tilde = None):

    y = np.exp(y_tilde)
    # only valid for current calc
    y = (y) - 9.333665e-01

    return y
def transf_Woo(data: pd.DataFrame = None):
    """
    Applies one of the implemented transfer frunctions to the given Dataset
    Returns the datafarme expanded by a colum which contains the new calculated y_tilde 
    eg. y_tilde = log_10(id) would be a reasonable transfer function
    """
    # removes faulty values which would cause errors for the given calc
    data = data[(data['v1'] != 0) & ((data['v1'] * data ['id']) <= 0)]
    
    y = data.loc[:,'id']
    v1 = data.loc[:,'v1']
    k = -1 
    y_tilde = np.log10((y/v1)*k)

    data_aug = data.copy()
    data_aug['id'] = y_tilde
    data_aug.columns = ['v1','v2','v3','v4','y_tilde']
    print('shape data_aug: '+ str(data_aug.size))
    print('shape data: '+ str(data.size))

    return data_aug,data

def inv_transf_Woo (y_tilde,v1):
    y = 10**(y_tilde) * (-1) *  v1
    return pd.Series(y)

def transf_Thung(data: pd.DataFrame = None, yname = 'id', Vds = "Vlat21",improved_del = 'off',epsilon = 1E-3):
    """
    Applies one of the implemented transfer frunctions to the given Dataset
    Returns a datafarme where is y is replaced by the new calculated y_tilde and the original dataframe. 
    eg. y_tilde = log_10(id) would be a reasonable transfer function
    """
    if improved_del == 'on':
        data = replace_small_v1(data, yname, Vds = Vds, epsi = epsilon)
        data = data[(data[Vds] * data [yname]) <= 0]


    else:
        # --- removes faulty values which would cause errors for the given calc ---
        data = data[(data[Vds] != 0) & ((data[Vds] * data [yname]) <= 0)]

    y = data.loc[:,yname]
    Vds = data.loc[:,Vds]
    # --- actuall Thung transfer: ---
    k = -1 # necessary because ilat1 and v1 should always have oposing signs
    y_tilde = np.log((y/Vds)*k)

    data_aug = data.copy()
    data_aug[yname] = y_tilde
    data_aug.rename(columns={yname:'y_tilde'},inplace=True) 

    # data_aug["y_tilde"] = data_aug["y_tilde"]/48


    print('--- Performed Thung transfer: ---')
    print(data_aug.describe())
    print(data.describe())      

    return data_aug,data


def mx_replace_zeros(data: pd.DataFrame):
    V_epsi = 1e-3

    # --- constructing replacement df for v1 = zeros and super small v1 ---
    all_small_v1_df = data.copy()
    s = np.sign(data["ids"])
        # --- catching the very unlikely case that current is exactly zero  
    for i in range(len(s)):
        if s[i] == 0:
            raise Exception("ids == 0 is bad")
            # s[i] = 1
            
               
    all_small_v1_df['Vlat21'] = s*V_epsi

    print('--- zeros replacement df: ---')
    print(all_small_v1_df.describe())
        # --- overwrites all small v1 values with 1E03 or -1E03  
        
    df_no_small_v1 =  data.where(((data['Vlat21'] < -V_epsi ) | (data['Vlat21'] > V_epsi)),other= all_small_v1_df)

    ## replace zero ids values


    print('---- Dataframe after replacing super small Vlat21 values: ----')
    print(df_no_small_v1.describe())
    data = df_no_small_v1
    return data

def replace_small_v1(data: pd.DataFrame, yname, Vds = "Vlat21",epsi = 1E-03):
           
        # --- constructing replacement df for v1 = zeros and super small v1 ---
    all_small_v1_df = data.copy()
    s = np.sign(data[yname])
        # --- catching the very unlikely case that current is exactly zero  
    # for i in range(len(s)):
    #     if s[i] == 0:
    #         s[i] = 1
               
    all_small_v1_df[Vds] = s*epsi

    print('--- zeros replacement df: ---')
    print(all_small_v1_df.describe())
        # --- overwrites all small v1 values with 1E03 or -1E03  
        
    df_no_small_v1 =  data.where(((data[Vds] < -epsi ) | (data[Vds] > epsi)),other= all_small_v1_df)
    print('---- Dataframe after replacing super small Vlat21 values: ----')
    print(df_no_small_v1.describe())
    data = df_no_small_v1
    return data

def inv_transf_Thung (y_tilde,Vlat21):
    # --- will always predict ilat1 ---
    y = np.multiply(np.exp(y_tilde),Vlat21) * (-1)
    print(y.shape)
    return y

def transf_V (data: pd.DataFrame = None):
    data = data[(data['Vlat21'] != 0)]
    y = data.loc[:,'id']
    Vlat21 = data.loc[:,'Vlat21']
 
    y_tilde = y * 1/(np.exp((Vlat21)))
    data_aug = data.copy()
    data_aug['id'] = y_tilde
    data_aug.columns = ["Vlat21", "Vfglat1", "Vtglat1",'y_tilde']

    return data_aug

def inv_transf_V(y_tilde,Vlat21):
    y = np.exp(Vlat21) *  y_tilde
    return pd.Series(y)