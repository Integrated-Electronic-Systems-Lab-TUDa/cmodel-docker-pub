#import sys
#sys.path.insert(1,'c:\\Users\\jowil\\OneDrive\\Dokumente\\Uni\\MSc_4_Semester\\020_python\\021_symbolic_reg')

import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd 
import numpy as np

from matplotlib.cm import get_cmap
import matplotlib.cm
import matplotlib.colors
import qlat_highscores as qlhs
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import analyze_NN as NN_anal
import utility_func as uf
import transfer_func
import cmasher as cmr
from cycler import cycler

# 
def main():
    run_for_plot()
    return

def run_for_plot():

    path = 'id.tbl'
    read_in_col = 5
    my_y_name = 'y'
    my_scale = 'linear'
    volt_to_analyse = 'y'

    cu_voltages = {
        'v1' : 'sweep',
        'v2' : 'step',
        'v3' : -1.6,
        'v4' : 0.0 }
 
    # --- saving options ---
    save_folder = 'c:\\Users\\jowil\\OneDrive\\Dokumente\\Uni\\MSc_4_Semester\\040_Bilder\\0411_ploished_data_analysis\\'
    save_file_name = 'se1_y'+my_scale+'_v2.pdf'
    my_save_path = save_folder + save_file_name

    # --- changing current sign to get ilat2 
    df_all = uf.extract_all_data('./data/'+str(path),scale=True,factor= -1E03,
                                  yname = my_y_name ,columns = read_in_col,ftype='tbl')
    # --- appending v4 (vbg) values --- 
    #df_all['v4'] = np.zeros(shape =(len(df_all['v3']),1))
    print(df_all.describe())
    
    # volt,y_pred = calc_ypred_NN(sweep_grid=df_all.loc[:,['v1','v2','v3','v4']],model_type='ensembled')
    # y_pred = y_pred * (-1)

    #df_compressed = select_data_two_const(voltages = cu_voltages, df_full = df_all,epsilon = 0.05)
    
    #polt_simple(df_volt= df_compressed,y_plot = df_compressed.loc[:,['ilat2']].to_numpy(),voltage_to_plot='v1')
    #plot_scatter_with_color(df_volt= df_compressed,y_plot = df_compressed.loc[:,['y']].to_numpy(),volt_for_x_axis='v1',volt_for_color='v2',scale='log')
    #plot_table1D(voltages=cu_voltages,df_com=df_compressed,scale= my_scale,yname = my_y_name ,v_list=[-1.6,-0.8,0.8,1.6,0.0])
   # plot_pred1D(voltages=cu_voltages,df_com=df_compressed,scale=my_scale,
   #             mdl_One= False, mdl_Two=True,save_path = my_save_path,v_list=[-1.6,1.6,-0.8,0.8,0.0])
   # plot_Thesis_TCAD_vs_Reference(voltages=cu_voltages,df_com=df_compressed,scale=my_scale,
    #            mdl_One= True, mdl_Two=False,save_path = my_save_path,v_list=[-1.6,1.6,-0.8,0.8,0.0])
   
    #plot_historgramm(all_data=df_all,vol_to_plt='v1',apply_log=False)
   # plot_Thesis_histogramm(all_data=df_all,vol_to_plt=volt_to_analyse,scale=my_scale,save_path=my_save_path)
    
    #plot_backend_Thesis(backend_name='tanh',scale=my_scale,save_path=my_save_path)

    #voltage_grid = generate_sweep_grid(Sweep_V1_arr=np.linspace(-1.6,1.6,150+1),
    # Sweep_V2_arr = np.linspace(start = -1.6, stop = 1.6,num= 8+1),
    # const_V3=('v3',cu_voltages.get('v3')),const_V4=('v4',cu_voltages.get('v4')))
    # df_volt = voltage_grid

    # df_volt= df_volt.drop(columns=['v4'])
    #df_volt,y_pred = calc_ypred_NN(sweep_grid=voltage_grid,model_type='ensembled',
    #model_path_lin= './NN_models/Set3_NNlin_03',model_path_log='./NN_models/Set3_NNlog_05',modle_inputs=['v1','v2','v3'])
    #ilat2 = y_pred * -1
    
    #y_pred = qlhs.transfer_model(data=df_volt,transfer_model_name='Set3_transfer_Tune1')
    #y_pred = qlhs.combined_model(data=df_volt,combined_model_name='Set3_lin_Tune1')
    #y_pred = qlhs.ensembled_model(data=df_volt,model_name_lin='Set3_lin_Tune1',model_name_log='Set3_transfer_Tune1')
    #ilat2 = y_pred *-1
    #ilat2 = y_pred 

    #y_pred = qlhs.combined_model(data = df_volt,combined_model_name='id_015_03')
    

    #plot_model_alone(df_vol=df_volt, y_to_plot=ilat2,voltages_dict=cu_voltages, scale=my_scale,
    #               model_name= 'SR_Set3_ensembled',save_path=my_save_path)
    #curves = uf.read_in_TCAD_S2_csv('./plot_data/Set5_v3_for_VP_n1_6.csv')
    #plot_curves(curves=curves,save_path=my_save_path,scale=my_scale)
    return

def calc_ypred_NN(
    sweep_grid : pd.DataFrame,
    model_type = 'single_log',
    backend_type = 'simple',
    model_path_log = './NN_models/NN_gaus_log_03',
    model_path_lin = './NN_models/NN_gaus_07',
    modle_inputs = ['v1','v2','v3','v4'],
    model_path_backend = './NN_models/NN_backend_02',
    ):
    """ calculate the predicted y values from the given NN model. This method is only responsible for calling the right calc method, with the reight parameters.
    """
    if model_type == 'single_lin':
        y_pred = NN_anal.predict_with_single_NN_model(mdl_path=model_path_lin, mdl_type='lin', data=sweep_grid,modle_inputs=modle_inputs)
    elif model_type == 'single_log':
        y_pred = NN_anal.predict_with_single_NN_model(mdl_path=model_path_log, mdl_type='log', data= sweep_grid,modle_inputs= modle_inputs)
    elif model_type == 'ensembled':
        y_pred = NN_anal.predict_with_ensembled_NN_model(model_path_lin=model_path_lin,model_path_log=model_path_log, data= sweep_grid,
                                                     backend_type = backend_type,model_path_backend = model_path_backend,modle_inputs=modle_inputs)

    return sweep_grid, y_pred

def select_data_one_const(voltages, df_full = None, epsilon = 0.05):
    """ 
    Extracts only those values from dataframe where e.g. v2 and v4 have the specified const value. In this example v1 is the x1 voltage(the one to plot on x axis) 
    and 'v3' is irrelevant. 
    Params:            
           voltages: dict containing all the volatge assigments, e.g.:
            {'v1' : 'sweep',
            'v2' : 0.4,
            'v3' : 'step',
            }
           -> if Voltage is set as 'sweep' -> all values extracted -> plot on x1-axis
           -> if Voltage is set as 'step' unused/ no influence on plot -> all values extracted
           -> if Voltage is set as const -> only values where this volatge has the speciefied value (+-epsilon) are kept

            df_full : df containing the imput data
    Returns: 
        Compressed Data frame where only y values with const v2,v4 are used
    """
    # defines upper and lower bound -> values insinde those bounds are considerd to be euqual for the prpose of reducing dimensionality
    d = epsilon
    v_const = list()
    for v in voltages.keys():
        if type(voltages.get(v)) is float:
            v_const.append(v)
    va = v_const[0]
    ca = voltages.get(va)
  
    df_out = df_full[((df_full[va] <= ca+d) & (df_full[va] >= ca-d))]
    print('--- Dataset after setting one volatge as const: ---')
    print(df_out.describe())

    #df_test = df_out[((df_full['v2'] <= -1.6+d) & (df_full['v2'] >= -1.6-d))].copy()
    #print(df_test.head(30))
    
    return df_out

def select_data_two_const(voltages, df_full = None, epsilon = 0.005):
    """ 
    Extracts only those values from dataframe where e.g. v2 and v4 have the specified const value. In this example v1 is the x1 voltage(the one to plot on x axis) 
    and 'v3' is irrelevant. 
    Params:            
           voltages: dict containing all the volatge assigments, e.g.:
            {'v1' : 'sweep',
            'v2' : 0.4,
            'v3' : 'step',
            'v4' : 0.4}
           -> if Voltage is set as 'sweep' -> all values extracted -> plot on x1-axis
           -> if Voltage is set as 'step' unused/ no influence on plot -> all values extracted
           -> if Voltage is set as const -> only values where this volatge has the speciefied value (+-0.05) are kept

            df_full : df containing the imput data
    Returns: 
        Compressed Data frame where only y values with const v2,v4 are used
    """
    # defines upper and lower bound -> values insinde those bounds are considerd to be euqual for the prpose of reducing dimensionality
    d = epsilon
    v_const = list()
    for v in voltages.keys():
        if type(voltages.get(v)) is float:
            v_const.append(v)
    va = v_const[0]
    vb = v_const[1]
    ca = voltages.get(va)
    cb = voltages.get(vb)
  
    df_out = df_full[((df_full[va] <= ca+d) & (df_full[va] >= ca-d)) & ((df_full[vb] <= cb+d) & (df_full[vb] >= cb-d)) ]
    print(df_out.describe())

    #df_test = df_out[((df_full['v2'] <= -1.6+d) & (df_full['v2'] >= -1.6-d))].copy()
    #print(df_test.head(30))
    
    return df_out

def select_data_three_const(voltages, df_full = None):
    """ 
    Extracts only those values from dataframe where e.g. v2 and v4 have the specified const value. In this example v1 is the x1 voltage(the one to plot on x axis) 
    and 'v3' is irrelevant. 
    Params:            
           voltages: dict containing all the volatge assigments, e.g.:
            {'v1' : 'x1',
            'v2' : 0.4,
            'v3' : -0.4,
            'v4' : 0.4}
           -> if Voltage is set as x1 -> all values extracted -> plot on x1-axis
           -> if Voltage is set as 'None' unused/ no influence on plot -> all values extracted
           -> if Voltage is set as const -> only values where this volatge has the speciefied value (+-0.005) are kept

            df_full : df containing the imput data
    Returns: 
        Compressed Data frame where only y values with const v2,v4 are used
    """
    # defines upper and lower bound -> values insinde those bounds are considerd to be euqual for the prpose of reducing dimensionality
    d = 0.005
    v_const = list()
    for v in voltages.keys():
        if type(voltages.get(v)) is float:
            v_const.append(v)
    va = v_const[0]
    vb = v_const[1]
    vc = v_const[2]
    ca = voltages.get(va)
    cb = voltages.get(vb)
    cc = voltages.get(vc)

  
    df_out = df_full[((df_full[va] <= ca+d) & (df_full[va] >= ca-d)) & ((df_full[vb] <= cb+d) & (df_full[vb] >= cb-d)) & ((df_full[vc] <= cc+d) & (df_full[vc] >= cc-d)) ]
    print(df_out.describe())
    
    #gives us 145 data points where y should mainly depend on v1
    return df_out

def plot_table1D(voltages, df_com = None,scale = 'linear',
 v_list = [-1.6,1.6,-1.1,1.1,-0.6,0.6,-0.4,0.4,-0.2,0.2,-0.1],yname = 'y'):

    """
    Plots the table data. 
    Params:
       voltages: dict containing all the volatge assigments:
            'sweep' -> v_sweep, volatge is shown on x-axis.
            'step' -> v_step is set to each value in v_list once, and a sweep over v_sweep is plotted.
            float -> v_const: these two voltages are set constant for all plots.
            scale: ('linar' or 'log')
    """
    #--- reading out dict in order to perform labeling (correct names)
    v_const = list()
    for v in voltages.keys():
        if type(voltages.get(v)) is float:
            v_const.append(v)
        if voltages.get(v) == 'sweep':
            v_sweep = v
        if voltages.get(v) == 'step':
            v_step = v

    # --- constant values are only used for title in this func ---
    if len(v_const) == 2:
        va = v_const[0]
        vb = v_const[1]
        ca = voltages.get(va)
        cb = voltages.get(vb)
        title_str = 'i_lat2 Table Data for '+str(va)+ ' =' +str(ca)+'V and ' +str(vb) +' = '+str(cb)+'V'
    else:
        title_str = 'i_lat2 Table Data for custom settings'
        

    v_list.sort(reverse = True)
    d = 0.005
  
    # --- colors for plotting 

    cm_name = 'cmr.guppy'
    #--- number of colors
    n = len(v_list)

    colors = mpl.colormaps[cm_name](np.linspace(1, 0,n))


    #--- plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, axes = plt.subplots(figsize=(800*px,480*px))

    axes.set_prop_cycle(color=colors)
    
    for v_step_val in v_list:
        df_onLine = df_com[((df_com[v_step] <= v_step_val+d) & (df_com[v_step] >= v_step_val-d))]
        y_true = df_onLine[yname].to_numpy()
        x = df_onLine[v_sweep].to_numpy()
        cur_label = str(v_step) +' : ' + str(v_step_val)
        if scale == 'log':
            axes.set_yscale('log')
            y_true = np.abs(y_true)
        plt.plot(x, y_true, label = cur_label ,marker = '+', linestyle = 'dashed',linewidth = 1.5, markersize = 6)
    
    plt.ylabel('Strom in mA')
    plt.xlabel(str(v_sweep)+' in V')
    plt.legend()
    
    plt.title(title_str)
    plt.show()
    return

def plot_pred1D(voltages, df_com = None, mdl_One= False, mdl_Two = False,
                scale = 'linear', v_list = [-1.6,1.6,-1.1,1.1,-0.6,0.6,-0.4,0.4,-0.2,0.2,-0.1],
                save_path = None
                ):

    """
    Plots table data curves vs. pred curves, as specified
    Params:
       voltages: dict containing all the volatge assigments, e.g.:
            {'v1' : 'sweep',
            'v2' : 0.4,
            'v3' : 'step',
            'v4' : 0.4}
           -> if Voltage is set as sweep -> all values extracted -> plot on x1-axis
           -> if Voltage is set as step -> all values are extracted
           -> if Voltage is set as const -> only values where this volatge has the speciefied value (+-0.005) are kept
            v_list: list with volatge steps to plot as secondary sweep, if not set, all possible step values are plotted
            scale: ('linar' or 'log')
    """
    #--- reading out dict in order to perform prcice labeling (correct names)
    v_const = list()
    for v in voltages.keys():
        if type(voltages.get(v)) is float:
            v_const.append(v)
        if voltages.get(v) == 'sweep':
            voltage_x = v
        if voltages.get(v)== 'step':
            v_sec = v

    # --- constant values are only used for title in this func ---
    va = v_const[0]
    vb = v_const[1]
    ca = voltages.get(va)
    cb = voltages.get(vb)
    v_list.sort()
    d = 0.005
  
    #--- colors for plotting 
    cm_name = 'cmr.guppy'
    n = len(v_list)# number of colors
    if mdl_Two or mdl_One:
        n = n*2
    colors = mpl.colormaps[cm_name](np.linspace(0.0, 1.0,n))

    #--- plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, axes = plt.subplots(figsize=(1000*px,600*px))
    axes.set_prop_cycle(color=colors)

    for v_step_val in v_list:
        df_onLine = df_com[((df_com[v_sec] <= v_step_val+d) & (df_com[v_sec] >= v_step_val-d))]
        y_true = df_onLine['y'].to_numpy()
        x = df_onLine[voltage_x].to_numpy()
        cur_label = 'true ' +str(v_sec) +': ' + str(v_step_val)
        if scale == 'log':
            axes.set_yscale('log')
            y_true = np.abs(y_true)
        plt.plot(x, y_true, label = cur_label ,marker = '+', linestyle = 'dashed',linewidth = 1.0, markersize = 5)

        if mdl_One:
            model_name = 'SR_V7'
            df_X_OneLine = df_onLine.loc[:,['v1','v2','v3','v4']]
            
            y_pred = qlhs.ensembled_model(data=df_X_OneLine,model_name_lin='id_015_03',model_name_log='tuning160')
   
            # --- because we predict ilat1 but want to plot ilat2:
            i_lat2_pred = y_pred * (1)
            if scale == 'log':
               i_lat2_pred = np.abs(i_lat2_pred)
            cur_plabel = 'pred ' + str(v_sec) +': ' + str(v_step_val)

            plt.plot(x, i_lat2_pred, label=cur_plabel,marker = 'd', linestyle = 'dotted',linewidth = 1.0, markersize = 5)

        if mdl_Two:
            model_name = 'NN_ens_07_03 '
            mdl_path = './NN_models/NN_gaus_07'
            cur_p2label = 'pred02: ' + str(v_sec) +': ' + str(v_step_val)
            df_X_OneLine = df_onLine.loc[:,['v1','v2','v3','v4']]
            #y_pred02 = NN_anal.predict_with_single_NN_model(mdl_path = mdl_path,data=df_X_OneLine,mdl_type='lin')
            y_pred02 = NN_anal.predict_with_ensembled_NN_model(data=df_X_OneLine,backend_type='simple',model_path_lin='./NN_models/NN_linTune3',model_path_log='./NN_models/NN_logTune3')
            #y_pred02 = NN_anal.predict_with_ensembled_NN_model(data=df_X_OneLine,backend_type='simple',model_path_lin='./NN_models/NN_gaus_07',model_path_log='./NN_models/NN_gaus_log_03')
            i_lat2_pred02 = y_pred02 * (-1)
            if scale == 'log':
               i_lat2_pred02 = np.abs(i_lat2_pred02)
            # --- create correct label
            cur_plabel = 'pred ' + str(v_sec) +': ' + str(v_step_val)

            plt.plot(x, i_lat2_pred02, label=cur_p2label,marker = 'd',linestyle = 'dotted',linewidth = 1.0, markersize = 4)
        if ((mdl_One == False) & (mdl_Two == False)):
           model_name = 'table data' 
    plt.ylabel('Strom in mA')
    plt.xlabel(str(voltage_x)+' in V')
    plt.legend()
    title = 'i_lat2 Comparison for '+str(va)+ ' =' +str(ca)+'V and ' +str(vb) +' = '+str(cb)+'V '+'for '+model_name
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return

def plot_model_alone(df_vol,y_to_plot, voltages_dict : dict, scale = 'lin', model_name = 'my_Model',save_path = None):
    print('--- exectuing plot model alone ---')
    v_const = list()
    for v in voltages_dict.keys():
        if type(voltages_dict.get(v)) is float:
            v_const.append(v)
        if voltages_dict.get(v) == 'sweep':
            sweepV_name = v
        if voltages_dict.get(v)== 'step':
            stepV_name = v
    
    va = v_const[0]
    vb = v_const[1]
    ca = voltages_dict.get(va)
    cb = voltages_dict.get(vb)

    df_plot = df_vol
    df_plot['y'] = y_to_plot

    # --- margin by which sweept to can vary and still be considerd equal
    margin = 1E-05
    decimals = int(np.log10(margin) *-1)
    print(decimals)

    sweep_scd_values = df_vol[stepV_name].unique()
   
    # --- in order to avoid getting many unique values which differ only by e.g. 1E-10
    sweep_scd_values = np.around(sweep_scd_values,decimals=decimals)

    #--- colors for plotting 
    cm_name = 'cmr.guppy'
    n = np.size(sweep_scd_values)# number of colors
    print(n)
    colors = mpl.colormaps[cm_name](np.linspace(0.0, 1.0,n))

    #--- plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    pic_scale = 1.0
    fig, axes = plt.subplots(figsize=((pic_scale*900*px,pic_scale*600*px)))
    axes.set_prop_cycle(color=colors)

    for v_scd in sweep_scd_values:
        df_onLine = df_plot[((df_plot[stepV_name] <= v_scd+margin) & 
                            (df_plot[stepV_name] >= v_scd-margin))]
        x= df_onLine[sweepV_name]
        y= df_onLine['y']
        cur_label = str(stepV_name) +': ' + str(v_scd) +'V'
        if scale == 'log':
            axes.set_yscale('log')
            y = np.abs(y)
        plt.plot(x, y, label = cur_label ,marker = '+', linestyle = 'dashed',linewidth = 1.5, markersize = 5)

    plt.ylabel('Strom in mA')
    plt.xlabel(str(sweepV_name)+' in V')
    plt.legend()
    title = 'i_lat2 predictions with: ' +model_name +' for: '+str(va)+ ' = ' +str(ca)+'V and ' +str(vb) +' = '+str(cb)+'V'
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    return
    
def polt_simple(df_volt, y_plot,voltage_to_plot = 'v1',scale  = 'lin'):
    """ Plots x over y as scatter plot. 
        The voltage to plot must be a column name of the dataframe with the voltage Values
    """
    x = df_volt.loc[:,[voltage_to_plot]].to_numpy()
    

    #--- plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(1000*px,600*px))

    if scale == 'log':
               y_plot = np.abs(y_plot)

    plt.scatter(x,y_plot)
    if scale == 'log':
        plt.yscale('log')
    plt.ylabel('Strom in mA')
    plt.xlabel(str(voltage_to_plot)+' in V ')
    title = 'Ilat2 over v1 for Dataset 6' 
    plt.title(label =  title)
    plt.show()
    
    return

def plot_historgramm(all_data , vol_to_plt = 'v1', apply_log = False):
    v_plt_arr = all_data[vol_to_plt].to_numpy()
    if apply_log:
        v_plt_arr = np.abs(v_plt_arr)
        v_plt_arr = np.log10(v_plt_arr+1E-7)

    counts, bins = np.histogram(v_plt_arr,bins = 201)

    
    fig, ax = plt.subplots()    
    ax.stairs(counts, bins)
    #plt.xticks(np.linspace(min(v_plt_arr)-0.1, max(v_plt_arr)+0.1, 18))
    ax.set_title(str(vol_to_plt)+ ' Distribution for Dataset 1')
    ax.set_xlabel(' '+str(vol_to_plt) + ' in V ')
    #ax.set_xlim(-20,1)
    plt.show()
    return

def plot_backend(backend_name = 'tanh',scale = 'log'):
    yname = 'id'
    scale_factor = 1E03
    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4",yname]
    all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
    train, test  = uf.train_test_split(all_data,train_size=0.01,random_state=9)

    
    #data_aug,train = transfer_func.transf_Thung(train,yname=yname)
    ilat = train.loc[:,yname].to_numpy()

    print(ilat.shape)
    ilat = np.sort(ilat)
    
    #sigma_log = data_aug['y_tilde']
    
    if backend_name == 'tanh':
    # ---- tanh activation----
        x = ilat
        epsilon = 1e-40
        slope_scale = 4
        offset = 8
        alpha = 0.5*np.log10(np.square(x)+epsilon)
        beta = 0.5*(1+np.tanh(slope_scale*alpha+offset))
        
    elif backend_name == 'sigmoid_ilat':
    # --- sigmoid activation based on ilat ---
        offset = 1E-4
        slope_scale = np.array([[1e05]])
    
        x = np.square(ilat)
        exponet = (-slope_scale)*(x-offset)
        beta = (1/(1+np.exp(exponet)))
        exponet = exponet.transpose()
        beta = beta.transpose()
    
    y = beta
    
    fig, ax = plt.subplots()
    plt.plot(ilat,y)
    ax.set_xscale(scale)
    #ax.set_ybound(-1000.0,1000.0)
    #ax.set_yscale('log')
    ax.set_xlabel('ilat')
    ax.set_ylabel('beta')
    plt.title('offset = '+str(offset)+' and scale = '+str(slope_scale))
    #plt.title('Values of a in 1/1+exp(a)')
    plt.show()
    return
   
def generate_sweep_grid(
    Sweep_V1_name = 'v1',Sweep_V1_arr = np.linspace(start =-1.6, stop = 1.6,num = 32+1),
    Sweep_V2_name = 'v2', Sweep_V2_arr = np.linspace(start = -1.6, stop = 1.6,num= 8+1),
    const_V3 = ('v3',1.6),
    const_V4 = ('v4',1.6)
):
    """ Generating a sweep grid for the 4 voltages. 
        Args:
            Sweep_V1_name = main sweep voltgae
            Sweep_V1_arr  = arr_containing all the sweep points. default = np.linspce(-1.6,1.6,33)
            Sweep_V2_name = secondary sweep voltage
            Sweep_V2_arr = arr containing volatge point for v2
            const_V3 = (name,value), default = (v3,1.6)
            const_V4 = (name,value), default = (v3,1.6)
        Returns:
            pd.Dataframe with the columns ['v1','v2','v3','v4'] and the sweeping grid as values
    """
    v1_values,v2_values = np.meshgrid(Sweep_V1_arr,Sweep_V2_arr)
    v1_values = v1_values.flatten()
    v2_values = v2_values.flatten()

    df_out = pd.DataFrame()
    df_out['x1'] = v1_values
    df_out['x2'] = v2_values
    df_out['x3'] = np.ones(shape = len(v1_values)) * const_V3[1]
    df_out['x4'] = np.ones(shape = len(v1_values)) * const_V4[1]
    df_out.columns = [Sweep_V1_name,Sweep_V2_name,const_V3[0],const_V4[0]]
    df_out = df_out[['v1','v2','v3','v4']]
    #print(df_out.head(40))
    return df_out

def plot_scatter_with_color(df_volt, y_plot,volt_for_x_axis = 'v1',volt_for_color ='v2',legend_type = 'colorbar',scale = 'lin'):
    """ Plots x over y as scatter plot. 
        The voltage to plot must be a column name of the dataframe with the voltage Values
    """
    """ Create a scatter plot where each dot is colored. 
        The color indicates a third varaibale (e.g. 'v2' values for plotting ilat2 over v1)
    """
    x = df_volt.loc[:,[volt_for_x_axis]].to_numpy()
    c = df_volt.loc[:,[volt_for_color]].to_numpy()
    c_max = np.round(np.max(c),1)
    c_min = np.round(np.min(c),1)
    x_name = rename_volatges(volt_for_x_axis)
    c_name = rename_volatges(volt_for_color)
    
    #--- colors for plotting 
    cm_name = 'cmr.guppy'
    #n = len(y_plot)# number of colors
    n = len(y_plot)
   # colors = mpl.colormaps[cm_name](np.linspace(0.1, 0.9,n))

    #--- plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    # A4 Seitenbreite: 
    fig, ax = plt.subplots(figsize=(900*px,600*px))

    s = np.ones(len(y_plot))*25

    if scale == 'log':
        y_plot = np.abs(y_plot)

    scatter = ax.scatter(x = x,s=s,y = y_plot, c = c,cmap=cm_name,vmin=-1.6,vmax=1.6,marker='x',linewidth = 1.5)
    
    
    if legend_type == 'discrete':
        #--- Produce a legend for the ranking (colors). Even though there are many different
        #--- v2 values, we only want to show 5 of them in the legend.
        legend1 = ax.legend(*scatter.legend_elements(num = [-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]),
                        loc="lower right", title="V2")
        ax.add_artist(legend1)
    elif legend_type == 'colorbar':
        #--- create an Axes on the right side of ax. The width of cax will be 5%
        #--- of ax and the padding between cax and ax will be fixed at 0.05 inch.
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = matplotlib.colors.Normalize(vmin=-1.6,vmax=1.6)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm_name), ax=ax,label = c_name+' in V')
        

    if scale == 'log':
        plt.yscale('log')

    plt.ylabel('Strom in mA')
    plt.xlabel(x_name+' in V ')
    title = 'Ilat2 values for Vtg = 1.6V' 
    plt.title(label =  title)
    ax.grid(True,alpha = 0.3)
    plt.show()
    
    

    return
def rename_volatges(name_in: str,tex = True):
    """ Aestetic naming. Returns nice Voltage names given the internal names. 
    """
    if (name_in == 'v1') or (name_in =='V1'):
        name_out = 'Vlat21'
        if tex: name_out = 'V\textsubscript{lat21}= '
    elif (name_in == 'v2') or (name_in =='V2'):
        name_out = 'Vfg'
        if tex: name_out = 'V\textsubscript{FG}= '
    elif (name_in == 'v3') or (name_in =='V3'):
        name_out = 'Vtg'
        if tex: name_out = 'V\textsubscript{TG}= '
    elif (name_in == 'v4') or (name_in =='V4'):
        name_out = 'Vbg'
        if tex: name_out = 'V\textsubscript{BG}= '
    else:
        name_out = name_in
    return name_out

def plot_curves(curves: pd.DataFrame,scale = 'lin',save_path = None):
    """fuction for plotting a number of Curves
        Args:
            Curves: Dataframe that contains all the cuvres to be plotted. Must be of the order of: X1,Y1,X2,Y2,X3,Y3,....
            If the curves do not have the same number of Datapoints, the must be filled up with nan.
    """
    n_curves = int (len(curves.columns) /2)
    
            #--- colors for plotting 
    cm_name = 'cmr.guppy'
    colors = mpl.colormaps[cm_name](np.linspace(0.0, 1.0,n_curves))

    #--- plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    pic_scale = 1.0
    fig, axes = plt.subplots(figsize=((pic_scale*900*px,pic_scale*600*px)))
    axes.set_prop_cycle(color=colors)

    for k in range(n_curves):
        x= curves.iloc[:,[2*k]].dropna()
        x = x.apply(lambda x: x-1.6)
        y = curves.iloc[:,[2*k+1]].dropna()
        cur_label = str(curves.columns[2*k])[:-2]
        if scale == 'log':
            axes.set_yscale('log')
            y = np.abs(y)
        plt.plot(x, y, label = cur_label ,marker = '+', linestyle = 'dashed',linewidth = 1.5, markersize = 5)

    plt.ylabel('Strom in mA')
    plt.xlabel('Vlat2 in V')
    plt.legend()
    title = 'TCAD Set5 Reference: Ilat2 for VP = -1.6V '
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    main()

def plot_Thesis_histogramm (all_data , vol_to_plt = 'v1', scale = 'lin',save_path = None):

    epsilon = 0.0
    n_bins = 401
    set_title = False
    x_label = r' I\textsubscript{lat21} (mA)'
    #x_label = (r' log\textsubscript{10} (I\textsubscript{lat21})')
    y_label = r'\# samples'
    log_y = True
    
    v_plt_arr = all_data[vol_to_plt].to_numpy()
    
    if scale == 'log':
        v_plt_arr = np.abs(v_plt_arr)
        #v_plt_arr = np.log10(v_plt_arr+epsilon)

    counts, bins = np.histogram(v_plt_arr,bins = n_bins)

    # --- fonts ans size for plotting:
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "charter",
    'font.size': 9, # in order to get fontsize of 11 for scaling to 0.8 textwidth
    'figure.dpi':120
    })
    plt.rc('legend',fontsize=9*(11/11)) # using a size in points

    # --- plotting params---
    size_controll = 0.46
    size = 'fig'
    cm = 1/2.54
    w_cm = 17.5655 *size_controll
    figsize_std = (w_cm *cm, w_cm*cm * (2/3))

    # --- generating new figure ----
    fig, ax = plt.subplots(figsize=figsize_std)
    ax.set_facecolor('#ffffffff')    
    ax.stairs(counts, bins)
    if set_title:
        ax.set_title(str(vol_to_plt)+ ' Distribution for Dataset 1')
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    #x_ticks = np.arange(-1.6,2.0,step=0.4)
    #ax.set_xticks(x_ticks,minor=False)
    if log_y: plt.yscale('log')
    #plt.xscale('log')
    ax.minorticks_on()
    #ax.set_xlim(-20,1)
    
    if size == 'fig':
        fig.tight_layout(pad=0.2)
    #else:
    #    plt.subplots_adjust(bottom=0.125,top=0.975,right=0.975,left=0.125) 
    #plt.legend(loc = 'upper center',ncol = 3)
    #plt.legend(loc='center left')
    if save_path is not None:
        plt.savefig(save_path,format='pdf',dpi= 120)
    plt.show()
    return
def plot_Thesis_TCAD_vs_Reference(voltages, df_com = None, mdl_One= False, mdl_Two = False,
                scale = 'linear', v_list = [-1.6,1.6,-1.1,1.1,-0.6,0.6,-0.4,0.4,-0.2,0.2,-0.1],
                save_path = None, compute_fresh = True, size_controll: float = 0.8, size = 'fig'
                ):

    """
    Plots table data curves vs. pred curves, as specified
    Params:
       voltages: dict containing all the volatge assigments, e.g.:
            {'v1' : 'sweep',
            'v2' : 0.4,
            'v3' : 'step',
            'v4' : 0.4}
           -> if Voltage is set as sweep -> all values extracted -> plot on x1-axis
           -> if Voltage is set as step -> all values are extracted
           -> if Voltage is set as const -> only values where this volatge has the speciefied value (+-0.005) are kept
            v_list: list with volatge steps to plot as secondary sweep, if not set, all possible step values are plotted
            scale: ('linar' or 'log')
    """
    #--- reading out dict in order to perform prcice labeling (correct names)
    v_const = list()
    for v in voltages.keys():
        if type(voltages.get(v)) is float:
            v_const.append(v)
        if voltages.get(v) == 'sweep':
            voltage_x = v
        if voltages.get(v)== 'step':
            v_sec = v

    # --- constant values are only used for title in this func ---
    va = v_const[0]
    vb = v_const[1]
    ca = voltages.get(va)
    cb = voltages.get(vb)
    v_list.sort()
    d = 0.005
  
    #--- colors for plotting 
    cm_name = 'cmr.guppy'
    n = len(v_list)# number of colors
    colors_Ref = mpl.colormaps[cm_name](np.linspace(0.0, 1.0,n))
   
    cc = (cycler(color=colors_Ref) *
        cycler(linestyle=['-', '--']))
    # --- fonts ans size for plotting:
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "charter",
    'font.size': (11), # in order to get fontsize of 11 for scaling to 0.8 textwidth
    'figure.dpi':120
    })
    plt.rc('legend',fontsize=7) # using a size in points

    # --- plotting params---
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    cm = 1/2.54
    w_cm = 17.5655 *size_controll
    figsize_std = (w_cm *cm, w_cm*cm * (2/3))
    fig, axes = plt.subplots(figsize=figsize_std)
    axes.set_facecolor('#ffffffff')
    axes.set_prop_cycle(cc)

    if size == 'plot':
        # --- increasing size by 15% to make romm for labels ---
        plt.gcf().set_size_inches(w_cm *cm*(1/0.85), w_cm*cm * (2/3)*(1/0.85))
    
   
    # --- actual plotting ---

    for v_step_val in v_list:
        df_onLine = df_com[((df_com[v_sec] <= v_step_val+d) & (df_com[v_sec] >= v_step_val-d))]
        y_true = df_onLine['y'].to_numpy()
        x = df_onLine[voltage_x].to_numpy()
        cur_label = 'TCAD: ' + r'V\textsubscript{FG}= ' + str(v_step_val)+'V'
        if scale == 'log':
            axes.set_yscale('log')
            y_true = np.abs(y_true)
        plt.plot(x, y_true, label = cur_label ,marker = 'none',
                 linestyle = '--',fillstyle = 'none', linewidth = 1.0, markersize = 7)

        if mdl_One:
            model_name = 'SR_V7'
            df_X_OneLine = df_onLine.loc[:,['v1','v2','v3','v4']]
            # --- create correct label
            cur_p2label = 'SR set1: ' + r'V\textsubscript{FG}= ' + str(v_step_val)+'V'
            if compute_fresh == True:
                volt_grid = generate_sweep_grid(Sweep_V1_arr=np.linspace(start=-1.6,stop=1.6,num = 33),
                Sweep_V2_arr=[v_step_val],const_V3=(va,ca),const_V4=(vb,cb))
                x = volt_grid[voltage_x].to_numpy()
                y_pred = qlhs.ensembled_model(data=volt_grid,model_name_lin='id_015_03',model_name_log='tuning160')

            elif compute_fresh == False:
                y_pred = qlhs.ensembled_model(data=df_X_OneLine,model_name_lin='id_015_03',model_name_log='tuning160')
   
            # --- because we predict ilat1 but want to plot ilat2:
            i_lat2_pred = y_pred * (1)
            if scale == 'log':
               i_lat2_pred = np.abs(i_lat2_pred)
            

            plt.plot(x, i_lat2_pred, label=cur_p2label,marker = 'o',linestyle = 'none',linewidth = 1.0, 
                     markersize = 4,markeredgewidth=1.0, fillstyle = 'none')

        if mdl_Two:
            model_name = 'NN_ens_Tune33 '
            mdl_path = './NN_models/NN_gaus_07'
            # --- create correct label
            cur_p2label = 'NN set1: ' + r'V\textsubscript{FG}= ' + str(v_step_val)+'V'

            df_X_OneLine = df_onLine.loc[:,['v1','v2','v3','v4']]
            
            if compute_fresh == True:
                volt_grid = generate_sweep_grid(Sweep_V1_arr=np.linspace(start=-1.6,stop=1.6,num = 32),
                Sweep_V2_arr=[v_step_val],const_V3=(va,ca),const_V4=(vb,cb))
                y_pred02 = NN_anal.predict_with_ensembled_NN_model(data=volt_grid,backend_type='simple',model_path_lin='./NN_models/NN_linTune3',model_path_log='./NN_models/NN_logTune3')
                x = volt_grid[voltage_x].to_numpy()
            elif compute_fresh == False:
                #y_pred02 = NN_anal.predict_with_single_NN_model(mdl_path = mdl_path,data=df_X_OneLine,mdl_type='lin')
                y_pred02 = NN_anal.predict_with_ensembled_NN_model(data=df_X_OneLine,backend_type='simple',model_path_lin='./NN_models/NN_linTune3',model_path_log='./NN_models/NN_logTune3')
                #y_pred02 = NN_anal.predict_with_ensembled_NN_model(data=df_X_OneLine,backend_type='simple',model_path_lin='./NN_models/NN_gaus_07',model_path_log='./NN_models/NN_gaus_log_03')
            
            i_lat2_pred02 = y_pred02 * (-1) 
            if scale == 'log':
               i_lat2_pred02 = np.abs(i_lat2_pred02)

            
            plt.plot(x, i_lat2_pred02, label=cur_p2label,marker = 'o',linestyle = 'none',linewidth = 1.0, 
                     markersize = 4,markeredgewidth=1.0, fillstyle = 'none')


        if ((mdl_One == False) & (mdl_Two == False)):
           model_name = 'table data' 
    plt.ylabel(r'I\textsubscript{lat21} (mA)')
    if scale == 'log':
        plt.ylim(top = 10, bottom = 1E-20)
    plt.xlabel(r'V\textsubscript{lat21}  (V)')
    plt.legend(facecolor='#ffffffff', framealpha=0.7)
    
    #title = 'i_lat2 Comparison for '+str(va)+ ' =' +str(ca)+'V and ' +str(vb) +' = '+str(cb)+'V '+'for '+model_name
    #plt.title(title)
    if save_path is not None:
        if size == 'fig':
            fig.tight_layout(pad=0.2)
        else:
            plt.subplots_adjust(bottom=0.125,top=0.975,right=0.975,left=0.125)
        

        plt.savefig(save_path,format='pdf',dpi= 120)
        # ,bbox_inches = 'tight'
    
    plt.show()
    return
def plot_backend_Thesis(backend_name = 'tanh',scale = 'log',save_path = None):

    set_title = False
    x_label = r' I\textsubscript{log} (mA)'
    #x_label = (r' log\textsubscript{10} (I\textsubscript{lat21})')
    y_label = r'$\beta$'


    yname = 'id'
    scale_factor = 1E03
    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4",yname]
    all_data[yname] = all_data[yname].apply(lambda x: x*scale_factor)
    train, test  = uf.train_test_split(all_data,train_size=0.01,random_state=9)    
    #data_aug,train = transfer_func.transf_Thung(train,yname=yname)
    ilat = train.loc[:,yname].to_numpy()

    print(ilat.shape)
    ilat = np.sort(ilat)
    
    #sigma_log = data_aug['y_tilde']
    
    if backend_name == 'tanh':
    # ---- tanh activation----
        ilat21 = (-1 * ilat)
        x = ilat21
        epsilon = 1e-40
        a = 4
        b = 8
        u = 0.5*np.log10(np.square(x)+epsilon)
        beta = 0.5*(1+np.tanh(a*u+b))

    y = beta
    
    # --- fonts ans size for plotting:
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "charter",
    'font.size': 9, # in order to get fontsize of 11 for scaling to 0.8 textwidth
    'figure.dpi':120
    })
    plt.rc('legend',fontsize=9*(11/11)) # using a size in points

    # --- plotting params---
    size_controll = 0.46
    size = 'fig'
    cm = 1/2.54
    w_cm = 17.5655 *size_controll
    figsize_std = (w_cm *cm, w_cm*cm * (2/3))

    # --- generating new figure ----
    fig, ax = plt.subplots(figsize=figsize_std)
    ax.set_facecolor('#ffffffff')

    # --- actual plotting
    
    #x_minor = np.array([10])
    plt.plot(ilat21,y,linewidth = 0.8, color = 'tab:blue')

    # --- scaling and such ---
    ax.set_xscale(scale)
    #ax.set_ybound(-1000.0,1000.0)
    #ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if scale == 'log':
        plt.xlim(right = 2E0, left = 1E-15)
        x_minor = [1,1E-1,1E-2,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9,1E-10,1E-11,1E-12,1E-13,1E-14,1E-15]
        #x_major = ['','',1E-2,'','',1E-5,'','',1E-8,'','',1E-11,'','',1E-14,'']
        ax.set_xticks(x_minor,minor = True)
        plt.setp(plt.gca().get_xminorticklabels(), visible=False)
        #ax.set_xticklabels(labels=x_major)
        plt.axvspan(0.0027, 0.0375, color='tab:orange', alpha=0.5)
        plt.plot([1E-2],[0.5],marker = 'x',linestyle = 'none',linewidth = 1.5, 
                     markersize = 6,markeredgewidth=1.0,color = 'tab:red',fillstyle = 'none')

    if scale == 'lin':
        ax.minorticks_on()
    y_ticks = np.arange(0,1.1,step=0.25)
    ax.set_yticks(y_ticks,minor=False)
    
    #ax.minorticks_on()

    # #plt.title('offset = '+str(a)+' and scale = '+str(b))
    # #plt.grid(which='major',axis='y',linewidth = 0.4)
    # #plt.grid(which='major',axis='x',linewidth = 0.4)
    plt.grid(linewidth = 0.4)
    # ax.grid()
    if size == 'fig':
        fig.tight_layout(pad=0.2)
    #else:
    #    plt.subplots_adjust(bottom=0.125,top=0.975,right=0.975,left=0.125) 
    #plt.legend(loc = 'upper center',ncol = 3)
    if save_path is not None:
        plt.savefig(save_path,format='pdf',dpi= 120)
    plt.show()
    return