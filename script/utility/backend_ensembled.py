import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- method defining different ensembleing backends (how to mathematically ensemble two pretrained models)
# --- is used to chain a model making good predictions for very small currents (log model) 
# --- together with a model making good predictions for larger currents (lin model)

def ensembled_backend(y_pred_lin,test_data = None,sigma_log = None,y_pred_log = None,backend_name = 'tanh'):
    """Computes an combined prediction utilizing the predictions made by the linear model and the predictions 
        made be the logarithmic model

    Args:
     y_pred_lin: current prectiction (output) of the current best linear model given train
     y_pred_log: current prectiction (output) of the current best logarithmic model given train
     test: the test data frame 
     sigma_log: the logarithmic sigma prediction (output) of the current best log model 
                using the adjusted thung transfer
     backend_name: str: best backend -> 'tanh'
    Returns:
       i_pred: (numpy Nx1 mat) the best current guess calculated using the two model outputs
    """
    if y_pred_log is None:
        v1 = test_data.loc[:,['Vlat2']].to_numpy()
        y_pred_log = np.multiply(np.exp(sigma_log),v1) 
    
    if backend_name == 'sigmoid':
    # ---- sigmoid activation----
        sigma = np.exp(sigma_log)
        offset = (-5)
        slope_scale = 10
        beta = (1/(1+np.exp((-slope_scale)*(sigma-offset))))
        alpha = 1-beta

    elif backend_name == 'sigmoid_ilat':
        # --- sigmoid activation based on ilat ---
        offset_current_scale = 1E-02
        offset = offset_current_scale**2
        slope_scale = 1E06
        x = np.square(y_pred_log)
        beta = (1/(1+np.exp((-slope_scale)*(x-offset))))
        alpha = 1-beta

    elif backend_name == 'tanh':
    # ---- this is the backend that peformed best ----
    # ---- tanh activation --- 
        x = y_pred_log
        epsilon = 1e-40
        slope_scale = 4
        offset = 8
        k_var = 0.5*np.log10(np.square(x)+epsilon)
        beta = 0.5*(1+np.tanh(slope_scale*k_var+offset))
        alpha = 1-beta

    
    elif backend_name == 'i_power':
        scaler2 = MinMaxScaler(feature_range = (-1,1))
        beta = scaler2.fit_transform(y_pred_lin)
        beta = np.power(beta,4)
        alpha = 1-beta
    else: 
    # --- alpha is one if sigma_log is greatly neative -> current is really small
    # # --- alpha is zero if sigma_log is close t0o zero or positive -> current is high
        scaler = MinMaxScaler(feature_range = (0,1))
        # --- squished sigma -> transforms sigma to be in the range of 0 - ca. 6
        squished_sgima = np.log2((-1)*sigma_log)
        alpha = scaler.fit_transform(squished_sgima)
    # ---- clac of i_pred ----
    i_pred = (1-alpha) * y_pred_lin + alpha * y_pred_log
    return i_pred