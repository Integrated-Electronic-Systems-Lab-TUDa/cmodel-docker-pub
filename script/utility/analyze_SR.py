import feyn
import numpy as np
import pandas as pd
import utility_func as uf
import sklearn.metrics as skm 
import visualize_results as visres
import qlat_highscores as qlhs
import transfer_func as transf
import backend_ensembled as back

def main():
    #analyze_single_SR(full_name = 'Set2_transferTune28',scale_ypred = 1, use_standart_data=False,
    #                transfer_model=True,combined_model=False,error_trsh = 0.025,PE_trsh = 0.1, set_v4_null=True)
    analyze_ensembled_SR(model_name_log = name_list_SR('SRs6')[1] ,model_name_lin=name_list_SR('SRs6')[0] ,test_all=True,use_data='set2'
                         ,scale_ypred=1,set_v4_null=False)
    #full_name_list = generate_name_list(start=100,end=196)
    #analyze_many_SR(full_name_list=full_name_list,save_path='./Qlattice/models/session_08_metrics')
    #analyze_charge_model(full_name='model_qlat1_01',charge_name='qlat1')
    return

def analyze_single_SR(
    full_name = 'tuning62',
    scale_ypred: float = 1,
    transfer_model = True,
    combined_model = False,
    gen2 = False,
    error_trsh = 0.05,
    PE_trsh = 0.3,
    use_standart_data = True,
    set_v4_null = False,

):
    if use_standart_data == False:
        yname = 'id'
        all_data = pd.read_table("./data/set2_lat2_m9.tbl")
        all_data.columns =["v1","v2","v3",yname]
        all_data['v4'] = np.zeros(shape =(len(all_data['v3']),1))
        all_data[yname] = all_data[yname].apply(lambda x: x*-1E03)
    else:
        all_data = pd.read_table("./data/id.tbl")
        all_data.columns =["v1","v2","v3","v4","id"]
        if set_v4_null:
                voltages =  {
                'v1' : None,
                'v2' : None,
                'v3' : None,
                'v4' : 0.0 }
                all_data = visres.select_data_one_const(voltages=voltages,df_full=all_data,epsilon=0.001)

        all_data["id"] = all_data["id"].apply(lambda x: x*1E03)

    if gen2:
        y_pred = qlhs.combined_model(data = all_data,combined_model_name = full_name, gen2 = gen2)
    elif transfer_model:
        y_pred = qlhs.transfer_model(data = all_data,transfer_model_name= full_name)
    elif combined_model:
        y_pred = qlhs.combined_model(data = all_data,combined_model_name= full_name)
    else:
        y_pred = qlhs.id_model(all_data,'./Qlattice/models/'+full_name+'.json')
        

    
    y_pred = y_pred * scale_ypred

    y_pred = y_pred.reshape(-1,1)
    y_true = all_data['id'].to_numpy().reshape(-1,1)
    

    uf.print_all_metrics(y_true=y_true,y_pred=y_pred,full_name=full_name,
    error_trsh=error_trsh,PE_threshold=PE_trsh)
    
    return

def analyze_many_SR(
    full_name_list = ['tuning62','tuning160'],
    transfer_model = True,
    combined_model = False,
    gen2 = False,
    save_path = './Qlattice/models/session_08_metrics'
    
):
    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4","id"]
    all_data["id"] = all_data["id"].apply(lambda x: x*1E03)


    metrics = ['MAPE','sMAPE','TewDev','maxPE']
    n_of_metrics = len(metrics)
    res = np.zeros(shape=(len(full_name_list),n_of_metrics))
    

    for idx,full_name in enumerate(full_name_list):
        if gen2:
            y_pred = qlhs.combined_model(data = all_data,combined_model_name = full_name, gen2 = gen2)
        elif transfer_model:
            y_pred = qlhs.transfer_model(data = all_data,transfer_model_name= full_name)
        elif combined_model:
            y_pred = qlhs.combined_model(data = all_data,combined_model_name= full_name)
        else:
            y_pred = qlhs.id_model(all_data,'./Qlattice/models/'+full_name+'.json')

        y_pred = y_pred.reshape(-1,1)
        y_true = all_data['id'].to_numpy().reshape(-1,1)
        res[idx,0] =skm.mean_absolute_percentage_error(y_true,y_pred)
        res[idx,1] = uf.Smape(y_true=y_true,y_pred=y_pred)
        res[idx,2] = uf.tweedie_dev(y_true=y_true,y_pred=y_pred)
        y_errors,_,_ = uf.critical_values(y_true=y_true,y_pred=y_pred)
        res[idx,3] = np.max(y_errors['PE'])

    df_results = pd.DataFrame(res)
    df_results.columns = metrics
    df_results.to_csv(path_or_buf=save_path)
    return

def analyze_ensembled_SR(
    backend_type = 'simple',
    model_name_log = 'tuning160',
    model_name_lin = 'id_015_03',
    model_path_backend = None,
    test_all = False,
    use_data = 'set1',
    set_v4_null = False,
    scale_ypred: float = 1
):
    
    # --- loading train data -----
    # --- using only unseen data for testing/ analysing 
    if use_data == 'set2':
        yname = 'i_lat1'
        scale_factor = -1E03
        all_data = pd.read_table("./data/set6_lat2_m13.tbl")
        all_data.columns =["v1","v2","v3",yname]
        all_data['v4'] = np.zeros(shape =(len(all_data['v3']),1))
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        test_all = True
    elif use_data == 'set5':
        yname = 'i_lat1'
        scale_factor = -1E03
        all_data = pd.read_csv("./data/Set5_v2.csv")
        all_data.columns =["v1","v2","v3",'v4',yname]
        all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
        test_all = True
    else:
        yname = 'i_lat1'
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

        trainVal, test = uf.split_trainVal_test_set(all_data = all_data)

    if test_all:
    # --- for instances where I want to perform test on the full dataset
    # --- could give infalated scores
        test = all_data

    X_test = test.loc[:,['v1','v2','v3','v4']]
    y_true = test.loc[:,[yname]].to_numpy()
  
    # ---- loading a pretrained model ----

    y_pred_lin =  qlhs.combined_model(combined_model_name =model_name_lin,data=X_test) 
    y_log_pred = qlhs.transfer_model(transfer_model_name=model_name_log,data=X_test)

   #---------actual model evalutaion--------
    if backend_type == 'simple':
       y_pred = back.ensembled_backend(y_pred_lin=y_pred_lin,y_pred_log=y_log_pred,backend_name='tanh')
    else :
       print('backend name unkown for SR application')

    y_pred = np.reshape(y_pred,newshape=(-1,1))
   
    PE_trsh = 0.1
    error_trsh = 0.025
    full_name = str(model_name_lin +'  with  '+ model_name_log)

    uf.print_all_metrics(y_true=y_true,y_pred=y_pred,full_name=full_name,
    error_trsh=error_trsh,PE_threshold=PE_trsh)
    return

def generate_name_list(stub = 'tuning', start = 100, end = 196):
    name_list = list()
    for n in range(start,end):
        name_list.append( str(stub) + str(n))
    return name_list

def analyze_charge_model(
    full_name = 'tuning62',
    scale_ypred: float = 1,
    error_trsh = 0.05,
    PE_trsh = 0.3,
    use_standart_data = True,
    set_v4_null = False,
    charge_name = 'qtg'
    ):

    if use_standart_data == False:
        yname = charge_name
        all_data = pd.read_table("./data/dummy")
        all_data.columns =["v1","v2","v3",yname]
        all_data['v4'] = np.zeros(shape =(len(all_data['v3']),1))
        all_data[yname] = all_data[yname].apply(lambda x: x*1E15)
    else:
        yname = charge_name
        all_data = pd.read_table('./data/'+charge_name+'.tbl')
        all_data.columns =["v1","v2","v3","v4",yname]
        if set_v4_null:
                voltages =  {
                'v1' : None,
                'v2' : None,
                'v3' : None,
                'v4' : 0.0 }
                all_data = visres.select_data_one_const(voltages=voltages,df_full=all_data,epsilon=0.001)

        all_data[yname] = all_data[yname].apply(lambda x: x*1E15)

        y_pred = qlhs.id_model(all_data,'./Qlattice/models/'+full_name+'.json')
        

    
    y_pred = y_pred * scale_ypred
    y_pred = y_pred.reshape(-1,1)
    y_true = all_data[yname].to_numpy().reshape(-1,1)
    
    uf.print_all_metrics(y_true=y_true,y_pred=y_pred,full_name=full_name,
    error_trsh=error_trsh,PE_threshold=PE_trsh)
    
    return  
def name_list_SR(name = 'SRs2'):
     if name == 'SRs2': i = 4
     elif name == 'SRs3': i = 5
     elif name == 'SRs6': i = 6
     elif name == 'SRs7': i = 7
     else:
        raise NameError('Model name unkown')
     name_list_lin = ['./NN_models/Set2_NNlin_04','./NN_models/Set3_NNlin_03','./NN_models/Set6_m13_NNlin_01','./NN_models/Set7_NNlin_01'
                      ,'Set2_lin_Tune1','Set3_lin_Tune1','Set6_lin_Tune3','Set7_lin_Tune1']
     name_list_log = ['./NN_models/Set2_NNlog_04','./NN_models/Set3_NNlog_05','./NN_models/Set6_NNlog_10','./NN_models/Set7_NNlog_03',
                      'Set2_transfer_Tune1','Set3_transfer_Tune1','Set6_transfer_Tune1','Set7_transfer_Tune1']
     return (name_list_lin[i],name_list_log[i])
if __name__ == "__main__":
    main()