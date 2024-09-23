import sys
sys.path.insert(1,'c:\\Users\\jowil\\OneDrive\\Dokumente\\Uni\\MSc_4_Semester\\020_python\\021_symbolic_reg')

import pandas as pd
import feyn
import numpy as np
import sklearn.metrics as skm 
from sklearn.model_selection import train_test_split
import transfer_func as transf
import utility_func as uf
import add_gaussian_noise as add_gaus


def main():
    run_ql_Qmodel()
    return

def run_ql_Qmodel(random_seed = 9, save_path = './Qlattice/models/Qmdl_qfg_03.json',num_epochs = 100,
    starting_model = None,train_size = 0.2,scale = True, transfer = 'off',
    max_complexity = 50,
    sorting_crit = 'bic'):
    
    #hyperparams
    scale_factor = 1E15
    n_epochs = num_epochs
    random_seed = random_seed
    y_name = 'qfg'
    # --- option to use other dataset ---
    all_data = pd.read_table("./data/qfg.tbl")
    all_data.columns =["v1","v2","v3","v4",y_name]


    #possiblility of scaling 
    if scale == True:
        all_data[y_name] = all_data[y_name].apply(lambda x: x*scale_factor)

    
    print(all_data.describe())
    train,test  = train_test_split(all_data, train_size=train_size,random_state=random_seed)

    ql = feyn.QLattice(random_seed=random_seed)
    output_name = y_name
    if starting_model is not None:
        start_from_model = starting_model
    else: 
        start_from_model = None

    #--- execute autorun ---
    models = ql.auto_run(data =train,kind = 'regression', output_name=output_name,
    n_epochs=n_epochs,max_complexity=max_complexity,criterion = sorting_crit,
    starting_models=start_from_model)

    #--- saving and printing results ---
    best = models[0]
    best.save(save_path)

    print('qlat_R2:'+ str(best.r2_score(test)))
    print('qlat_MAE:'+str(best.mae(test)))
    print('qlat_MSE:'+str(best.mse(test)))
    # --- vol list: e.g. 'v1','v2','v3','v4' for original Dataset---
    vol_list = all_data.columns.values.tolist()
    vol_list.remove(y_name) # del current colum 
    y_pred = best.predict(test[vol_list])
    y_true = test[y_name].to_numpy().reshape(-1,1)
    print('qlat own clac MAE: '+str(skm.mean_absolute_error(y_true, y_pred)))
    print('qlat own calc R2 score:'+str(skm.r2_score(y_true,y_pred)))
    print('qlat own calc MAPE:'+str(skm.mean_absolute_percentage_error(y_true, y_pred)))
    print('qlat own clac Median AE : '+str(skm.median_absolute_error(y_true, y_pred)))

    y_pred_a = best.predict(all_data[vol_list])
    y_true_a = all_data[y_name].to_numpy().reshape(-1,1)

    r2_all = skm.r2_score(y_true_a,y_pred_a)
    MSE_all = skm.mean_squared_error(y_true_a,y_pred_a)
    print('Qlat own clac R2 all :'+str(r2_all))
    print('Qlat own clac MSE all :'+str(MSE_all))
    
    return (r2_all,MSE_all)

if __name__ == "__main__":
    main()

