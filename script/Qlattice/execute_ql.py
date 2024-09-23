import sys
# --- has to be adjusted to the relative path of the repo (021_symbolic_reg) if run on a different machine (not on Server)
# sys.path.insert(1,'../')
import config

import Run_qlattice
import feyn
import utility_func as uf
import numpy as np
import pandas as pd

data_path= 'data/'

def main():
    run_SR()

    #run_tuning(data= './data/set6_lat2_m13.tbl',path_prefix= 'Set6_transfer_',transfer= 'Thung')
    #run_tuning(data= './data/set7_lat2_m13.tbl',path_prefix= 'Set7_transfer_',transfer= 'Thung')

    # run_tuning(data= './data/set6_lat2_m13.tbl',path_prefix= 'Set6_lin_',transfer= 'off')
    #run_tuning(data= './data/set7_lat2_m13.tbl',path_prefix= 'Set7_lin_',transfer= 'off')
    
    return
def run_tuning(data:str = './data/set2_lat2_m9.tbl',path_prefix:str = 'Set2_transfer_v2_',transfer:str = 'off'):
    """ Methode for performing SR tuning. This methode performs a grid search and trains one model 
        for every hyperparameter combination specified internally.
    """
    param_grid = {
    'n_epochs': [300],
    'low_bound': [np.nan],
    'sigma_gaus': [np.nan],
    'max_complexity': [64],
    'train_size': [0.99],
    'transfer': [transfer],
    'start_mdl':[False],
    'sorting_crit':['bic']
    }
    
    param_product_list = uf.generate_param_product_list(param_grid=param_grid)
    df_out_parmas = pd.DataFrame(param_product_list)
    df_out_parmas.to_csv(path_or_buf='./Qlattice/models/'+path_prefix+'Tune_params')

    r2_MSE_all = list()
    i_offset = 2
    i = i_offset
    # --- for each possible Hyperparam combination 
    for one_dict in param_product_list:
        
        i = i+1
        print('------ running< Iteration: '+ str(i)+ '----------')
        lsave_path = './Qlattice/models/'+path_prefix+'Tune'+str(i)+'.json'
        print(lsave_path)

        if one_dict.get('start_mdl'):
            starting_models =[feyn.Model.load('./Qlattice/models/Set2_transfer4.json')]
        else:
            starting_models = None


        r2_MSE = Run_qlattice.run_ql(random_seed=i,
        num_epochs= one_dict.get('n_epochs'),
        bound_ilat_to= one_dict.get('low_bound'),
        sigma_gaus_aug= one_dict.get('sigma_gaus'),
        max_complexity= one_dict.get('max_complexity'),
        train_size = one_dict.get('train_size'),
        transfer=one_dict.get('transfer'),
        save_path=lsave_path,
        starting_model=starting_models,
        scale = True,
        sorting_crit = one_dict.get('sorting_crit'),
        use_other_data=True,
        path_of_other_Data= data
        )
        print('--- score of Iteration: '+ str(i) + '---')
        r2_MSE_all.append(r2_MSE)
        print (r2_MSE_all)
    print('--- final scoring: ---')
    
    df_out_results = pd.DataFrame()

    R2_MSE_arr = np.zeros(shape = (len(r2_MSE_all),3))

    for n in range(0,i-i_offset):
        R2_MSE_arr[n,0] = r2_MSE_all[n][0]
        R2_MSE_arr[n,1] = r2_MSE_all[n][1]
        R2_MSE_arr[n,2] = n+i_offset
        print(r2_MSE_all[n])
        print(param_product_list[n])
    df_out_results['mdl_num'] = R2_MSE_arr[:,2]
    df_out_results['R2' ] = R2_MSE_arr[:,0]
    df_out_results['MSE'] = R2_MSE_arr[:,1]
    print(df_out_parmas.head(10))
    print(df_out_results.head(10))
    
    df_out_results.to_csv(path_or_buf='./Qlattice/models/Transfer_Tune_01_res')

    return


def run_SR():
    """ Main methode for training new SR modles. 
    """
    r2_MSE_all = list()
    for i in range (0,1):
        print('------ running seed: '+ str(i)+ '----------')
        lsave_path = './Qlattice/models/Set2_transfer'+str(i)+'.json'
        r2_MSE = Run_qlattice.run_ql(random_seed=i,
                                    num_epochs= 300,
                                    bound_ilat_to= np.nan,
                                    sigma_gaus_aug= 1E-11,
                                    max_complexity= 100,
                                    train_size = 0.99,
                                    transfer='Thung',
                                    save_path=lsave_path,
                                    # starting_model=[feyn.Model.load('./Qlattice/models/Set2_transfer1.json')],
                                    starting_model=None,
                                    scale = True,
                                    use_other_data = True,
                                    # path_of_other_Data = './data/set2_lat2_m9.tbl',
                                    path_of_other_Data = data_path + 'rfet_tsim_64_m13_amp1.6_os40' + '/lat2_current.tbl',
                                    sorting_crit='bic'
                                    )
        r2_MSE_all.append(r2_MSE)
        print (r2_MSE_all)

    # for i in range (24,24):
    #     print('------ running Iteration: '+ str(i)+ '----------')
    #     lsave_path = './Qlattice/models/mult_id_runs_0'+str(i)+'.json'
    #     print(lsave_path)
    #     starting_models = [feyn.Model.load('./Qlattice/models/model_id_12.json'),feyn.Model.load('./Qlattice/models/mult_id_runs_015.json'),feyn.Model.load('./Qlattice/models/mult_id_runs_05.json'),feyn.Model.load('./Qlattice/models/mult_id_runs_04.json'),feyn.Model.load('./Qlattice/models/mult_id_runs_03.json'),feyn.Model.load('./Qlattice/models/mult_id_runs_01.json'),feyn.Model.load('./Qlattice/models/mult_id_runs_02.json')]
    #     r2_MSE = Run_qlattice.run_ql(random_seed=i,save_path=lsave_path,num_epochs=20,starting_model=starting_models,scale = True,train_size=0.99)
    #     r2_MSE_all.append(r2_MSE)
    #     print (r2_MSE_all)
    return
if __name__ == "__main__":
    main()