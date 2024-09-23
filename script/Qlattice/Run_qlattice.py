import sys
# sys.path.insert(1,'c:\\Users\\jowil\\OneDrive\\Dokumente\\Uni\\MSc_4_Semester\\020_python\\021_symbolic_reg')
sys.path.insert(1,'../')
sys.path.insert(1,'../utility')

import pandas as pd
import feyn
import numpy as np
import sklearn.metrics as skm 
from sklearn.model_selection import train_test_split
import transfer_func as transf
import utility_func as uf
import add_gaussian_noise as add_gaus
import kennard_stone as ks


def run_ql(random_seed = 9, 
            save_path = './Qlattice/test.json',
            num_epochs = 3,
            starting_model = None,
            train_size = 0.2,
            scale = True,
            transfer = 'off',
            bound_ilat_to = np.nan,
            sigma_gaus_aug = np.nan,
            max_complexity = 150,
            sorting_crit = None,
            use_other_data = False,
            path_of_other_Data = None):
    
    #hyperparams
    scale_factor = 1E3
    n_epochs = num_epochs
    random_seed = random_seed
    y_name = 'id'
    # --- option to use other dataset ---
    if use_other_data:
        all_data = all_data = pd.read_table(path_of_other_Data)
        all_data.columns =["v1","v2","v3",y_name]
        # --- because original data had ilat1, new data has ilat2 ---
        scale_factor = -1*scale_factor
    else: 
        all_data = pd.read_table("./data/id.tbl")
        all_data.columns =["v1","v2","v3","v4",y_name]


    #possiblility of scaling 
    if scale == True:
        all_data[y_name] = all_data[y_name].apply(lambda x: x*scale_factor)
    if (np.isnan(sigma_gaus_aug) == False):
        all_data = add_gaus.augment_train_with_noise(trainVal=all_data,num_copies=3, yname=y_name,sigma = sigma_gaus_aug,Keep_original=True)
    if (np.isnan(bound_ilat_to) == False):
        all_data = uf.bound_current(all_data=all_data,ilat_col = y_name,lower_bound=bound_ilat_to)
    if transfer == 'Thung':
        y_name = 'y_tilde'
        all_data,_ = transf.transf_Thung(all_data,improved_del='on')
    
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
    print('qlat own calc MAE: '+str(skm.mean_absolute_error(y_true, y_pred)))
    print('qlat own calc R2 score:'+str(skm.r2_score(y_true,y_pred)))
    print('qlat own calc MAPE:'+str(skm.mean_absolute_percentage_error(y_true, y_pred)))
    print('qlat own cal Median AE : '+str(skm.median_absolute_error(y_true, y_pred)))

    y_pred_a = best.predict(all_data[vol_list])
    y_true_a = all_data[y_name].to_numpy().reshape(-1,1)

    r2_all = skm.r2_score(y_true_a,y_pred_a)
    MSE_all = skm.mean_squared_error(y_true_a,y_pred_a)
    print('Qlat own clac R2 all :'+str(r2_all))
    print('Qlat own clac MSE all :'+str(MSE_all))
    
    return (r2_all,MSE_all)





def mx_run_ql(random_seed = 9, 
            save_path = './Qlattice/test.json',
            num_epochs = 3,
            starting_model = None,
            train_size = 0.2,
            scale = True,
            transfer = 'off',
            bound_ilat_to = np.nan,
            sigma_gaus_aug = np.nan,
            max_complexity = 150,
            sorting_crit = None,
            data = None,
            Xlabels = ["Vlat21", "Vfglat1", "Vtglat1"], 
            Ylabel = ["ids"],
            id = "empty",
            mdl_type = "",
            scaling = 1E3,
            splitting_algorithm="TF"):
    
    #hyperparams
    # scale_factor = scaling
    n_epochs = num_epochs
    random_seed = random_seed
    y_name = Ylabel[0]
    # --- option to use other dataset ---
    # if use_other_data:
    #     all_data = all_data = pd.read_table(path_of_other_Data)
    #     all_data.columns =["v1","v2","v3",y_name]
    #     # --- because original data had ilat1, new data has ilat2 ---
    #     scale_factor = -1*scale_factor
    # else: 
    #     all_data = pd.read_table("./data/id.tbl")
    #     all_data.columns =["v1","v2","v3","v4",y_name]

    all_data = data[Xlabels + Ylabel]
    all_data_raw = all_data.copy()

    # #possiblility of scaling 
    if scale == True:
        all_data[y_name] = all_data[y_name].apply(lambda x: x*scaling)
    if (sigma_gaus_aug != None):
        all_data = add_gaus.augment_train_with_noise(trainVal=all_data,num_copies=3, yname=y_name,sigma = sigma_gaus_aug,Keep_original=True)
    if (bound_ilat_to != None):
        all_data = uf.bound_current(all_data=all_data,ilat_col = y_name,lower_bound=bound_ilat_to)
    if transfer == 'Thung':
        y_name = 'y_tilde'
        all_data,_ = transf.transf_Thung(all_data,yname=Ylabel[0], improved_del='on', epsilon=1E-3)
    
    print(all_data.describe())


    if splitting_algorithm=="kennard_stone":

        test = pd.DataFrame()
        train = pd.DataFrame()


        if train_size == 1.0:
            ks_x_train = all_data[Xlabels]
            ks_y_train = all_data[y_name]


        else:
            ks_x_train, ks_x_test, ks_y_train, ks_y_test = ks.train_test_split(all_data[Xlabels], all_data[y_name],test_size=1-train_size)
            test[Xlabels] = ks_x_test
            test[y_name] = ks_y_test
            
            
        train[Xlabels] = ks_x_train
        train[y_name] = ks_y_train

    else:
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
    best.save(save_path + "/" + mdl_type + "/" + id)

    print('qlat_R2:'+ str(best.r2_score(test)))
    print('qlat_MAE:'+str(best.mae(test)))
    print('qlat_MSE:'+str(best.mse(test)))
    # --- vol list: e.g. 'v1','v2','v3','v4' for original Dataset---
    vol_list = all_data.columns.values.tolist()
    vol_list.remove(y_name) # del current colum 
    y_pred = best.predict(test[vol_list])
    y_true = test[y_name].to_numpy().reshape(-1,1)
    print('qlat own calc MAE: '+str(skm.mean_absolute_error(y_true, y_pred)))
    print('qlat own calc R2 score:'+str(skm.r2_score(y_true,y_pred)))
    print('qlat own calc MAPE:'+str(skm.mean_absolute_percentage_error(y_true, y_pred)))
    print('qlat own cal Median AE : '+str(skm.median_absolute_error(y_true, y_pred)))


    # y_pred_a = best.predict(all_data[vol_list])
    # y_true_a = all_data[y_name].to_numpy().reshape(-1,1)
    y_pred_test = best.predict(test[vol_list])
    y_true_test = test[y_name].to_numpy().reshape(-1,1)

    r2_test = skm.r2_score(y_true_test/scale,y_pred_test/scale)
    MSE_test = skm.mean_squared_error(y_true_test/scale,y_pred_test/scale)
    MAE_test = skm.mean_absolute_error(y_true_test/scale,y_pred_test/scale)

    print('Qlat own clac R2 all :'+str(r2_test))
    print('Qlat own clac MSE all :'+str(MSE_test))
    



    return (train, test, best, MAE_test, MSE_test,r2_test)



