from feyn import Model as fModel
import transfer_func as transf
import backend_ensembled as back

# --- methods for predicting with previously saved SR(Qlattice) Models

def id_model(data,path,model_inputs = ['v1','v2','v3','v4']):

    current_model = fModel.load(path)
    y_pred = current_model.predict(data.loc[:,model_inputs])
    return y_pred
def combined_model(combined_model_name,data, gen2 = False):

    if combined_model_name == 'id_015_03':
        y_pred_a = id_model(data,'./Qlattice/models/mult_id_runs_015.json')
        y_pred_e = id_model(data,'./Qlattice/models/er_for_015_id_03.json')
        y_pred = y_pred_a + y_pred_e
        if gen2:
            y_pred_gen2 = id_model(data,'./Qlattice/models/er_gen2_for_015_03_n02.json')
            y_pred = y_pred + y_pred_gen2

    elif combined_model_name == 'id_015_04':
        y_pred_a = id_model(data,'./Qlattice/models/mult_id_runs_015.json')
        y_pred_e = id_model(data,'./Qlattice/models/er_for_015_id_04.json')
        y_pred = y_pred_a + y_pred_e
        if gen2:
            y_pred_gen2 = id_model(data,'./Qlattice/models/er_gen2_for_015_04_n04.json')
            y_pred = y_pred + y_pred_gen2
    elif combined_model_name == 'id_12_02':
        y_pred_a = id_model(data,'./Qlattice/models/model_id_12.json')
        y_pred_e = id_model(data,'./Qlattice/models/er_for_12_id_02.json')
        y_pred = y_pred_a + y_pred_e
        if gen2:
            y_pred_gen2 = id_model(data,'./Qlattice/models/er_gen2_for_12_02_n01.json')
            y_pred = y_pred + y_pred_gen2
    elif combined_model_name == 'id_12_01':
        y_pred_a = id_model(data,'./Qlattice/models/model_id_12.json')
        y_pred_e = id_model(data,'./Qlattice/models/er_for_12_id_01.json')
        y_pred = y_pred_a + y_pred_e
        if gen2:
            y_pred_gen2 = id_model(data,'./Qlattice/models/er_gen2_for_12_01_n03.json')
            y_pred = y_pred + y_pred_gen2
    elif combined_model_name == 'id_015':
        y_pred_a = id_model(data,'./Qlattice/models/mult_id_runs_015.json')
        y_pred = y_pred_a 
    else:
        path = './Qlattice/models/' + str(combined_model_name)+'.json'
        y_pred = id_model(data,path)
    return y_pred
def transfer_model(data,transfer_model_name):
    print(transfer_model_name)
    y_pred = None
    if transfer_model_name == 'transf_id_01':
        data_aug = transf.const_transf(data)
        print(data_aug.describe())
        path = './Qlattice/models/transf_id_01.json'
        y_tilde_pred = id_model(data_aug,path)
        y_pred = transf.inv_const_trans(y_tilde_pred)
    else:
        path = './Qlattice/models/' + str(transfer_model_name)+'.json'
        y_tilde_pred = id_model(data,path)
        v1 = data.loc[:,'v1']
        y_pred = transf.inv_transf_Thung(y_tilde_pred,v1).to_numpy()
    return y_pred
def ensembled_model(data,model_name_lin,model_name_log,backend_name = 'tanh'):
    y_pred_log = transfer_model(data=data,transfer_model_name=model_name_log)
    y_pred_lin = combined_model(data = data, combined_model_name=model_name_lin)
    y_pred = y_pred = back.ensembled_backend(y_pred_lin=y_pred_lin,y_pred_log=y_pred_log,backend_name=backend_name)
    return y_pred