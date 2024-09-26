
import sys
import os
import config


import mongodb_driver
import numpy as np
import warnings
import feyn
import os
import pathlib
import subprocess
import glob
import shutil

import pandas as pd
import numpy as np
import sklearn.metrics as skm 
import utility_func as uf

import re
import NN.NN_Readout as Readout
from tensorflow import keras 
import NN.NN_utility as NN_u
from bson.objectid import ObjectId
import pytaurus_tools as pyt
import matplotlib.pyplot as plt
import matplotlib.patches as patches




class rfet_model:
    
    # name = None

    # I_model_pars = {"lin" : None, "log": None}


    # Q_model_pars = None


    # str_template = None

    # nn_mdl_dir = None
    # sr_mdl_dir = None


    # model_lin = None
    # model_log = None

    # trace_cds_inv = None


    # weight_offset= 8
    # weight_slope = 4


    def __init__(self, name, I_model, Q_model, template_file = "templates/verilogA_template_par.va", nn_mdl_dir="./NN_models/mx/", sr_mdl_dir="./Qlattice/models"):
        self.name = name
        self.nn_mdl_dir = nn_mdl_dir
        self.sr_mdl_dir = sr_mdl_dir

        mdb = mongodb_driver.mdb_connect("mdb_cmodel", "root", "tyjWbtU4DRnuXqdhK")
        mdb_col = mdb["NN"]

        i_mdl_lin = mdb_col.find_one({"_id" : ObjectId(I_model["lin"])})
        
        self.I_model_pars = {}
        self.I_model_pars["lin"] = i_mdl_lin

        i_mdl_log = mdb_col.find_one({"_id" : ObjectId(I_model["log"])})
        
        self.I_model_pars["log"] = i_mdl_log

        q_mdl = mdb_col.find_one({"_id" : ObjectId(Q_model)})

        self.Q_model_pars = q_mdl

        with open(template_file, "r") as in_file:
            self.str_template = in_file.read()

        
    def setEnsemblePars(self, offset, slope):
        self.weight_offset = offset
        self.weight_slope = slope


    def compile(self, path ="./va_test.va", lin_prescale = None,log_postscale=1, Q_scale = 1E15, significant_digits = 4):

        if "lin_prescale" in self.I_model_pars["lin"]:
            lin_prescale = int(self.I_model_pars["lin"]["lin_prescale"])
        else:
            if lin_prescale==None:
                lin_prescale= 1
                print("PRESCALE NOT FOUND!")

        str_other_params = ""
        str_other_params = str_other_params + '// --- Backend params ---\n'
        # str_other_params = str_other_params + 'parameter real  offset = 6.8;\n'
        # str_other_params = str_other_params + 'parameter real slope = 2.9;\n'
        str_other_params = str_other_params + f"real  offset = {self.weight_offset};\n"
        str_other_params = str_other_params + f"real slope = {self.weight_slope};\n"
        str_other_params = str_other_params + 'real epsilon = 1E-100;\n'


        str_backend = '// --- ensembeling backed (tanh)---\n'

#        str_backend = str_backend   + 'k = 0.5*log((((Ilat1_lin-Ilat1_log)/(log(Ilat1_lin) - log(Ilat1_log)))**2)+epsilon);\n' \
        str_backend = str_backend   + 'k = 0.5*log((Ilat1_log**2)+epsilon);\n' \
                                    + 'beta = 0.5*(1+tanh(slope*k+offset));\n' \
                                    + f'out = beta* Ilat1_lin * {1/lin_prescale} + (1-beta)*Ilat1_log;\n'

        # I model
        if ("mdl_arch" in self.I_model_pars["lin"]) and (self.I_model_pars["lin"]["mdl_arch"] == "SR"):
            # SR model
            print("Lin model is Symbolic Regression")


            # generate lin model
            i_mdl_lin = feyn.Model.load(self.sr_mdl_dir + "/mx/lin/"+ str(self.I_model_pars["lin"]["_id"]) )
            self.model_lin = i_mdl_lin
            str_eq_lin = f"\n // I MODEL\n" + "Ilat1_lin = " + str(i_mdl_lin.sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)") + ";\n"


            # generate log model
            if(self.I_model_pars["log"] != None) :
                print("log model exists")
                if ("mdl_arch" in self.I_model_pars["log"]) and (self.I_model_pars["log"]["mdl_arch"] == "SR"):
                    print("log model is SR")
                    i_mdl_log = feyn.Model.load(self.sr_mdl_dir + "/mx/log/"+ str(self.I_model_pars["log"]["_id"]) )
                    self.model_log = i_mdl_log

                    str_model_log =  f"\nsigma_log =  " + str(i_mdl_log.sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)") + ";\n"
                    str_model_log = str_model_log + "//--- Backtransfer ---\n" + "Ilat1_log = volt[0] *(-1)*exp(sigma_log);\n"
                
                else:
                    print("log model is NN")
                    path_log = self.nn_mdl_dir + "log/" + str(self.I_model_pars["log"]["_id"])
                    save_path_log = self.nn_mdl_dir + "../mx_weights/" + "log/" + str(self.I_model_pars["log"]["_id"])+'.hdf5'
                    w_log = Readout.save_weights(save_path = save_path_log,model_path =path_log)
                    Readout.check_if_saved_and_loaded_weights_are_equal(weights_saved=w_log,weights_path=save_path_log)

                    str_weights_log = self.generateWeights(save_path_log)

                    str_model_log = self.generateLoops(save_path_log,out="sigma_log")
                    
                    str_model_log = str_model_log + "//--- Backtransfer ---\n" + "Ilat1_log = volt[0] *(-1)*exp(sigma_log);\n"
            
            at_parameters = f"\n // PARAMETERS\n" + str_other_params

            at_i_model = f"\n // I MODEL\n" + str_eq_lin + str_model_log + str_backend

        else:
            # NN model
            print("lin model is Neural Network")

            path_lin = self.nn_mdl_dir + "lin/" + str(self.I_model_pars["lin"]["_id"])
            self.model_lin = keras.models.load_model(path_lin,custom_objects = { 'R2': NN_u.R2})
            save_path_lin = self.nn_mdl_dir + "../mx_weights/" + "lin/" + str(self.I_model_pars["lin"]["_id"])+'.hdf5'
            w_lin = Readout.save_weights(save_path = save_path_lin,model_path =path_lin)
            Readout.check_if_saved_and_loaded_weights_are_equal(weights_saved=w_lin,weights_path=save_path_lin)

            str_weights_lin = self.generateWeights(save_path_lin, B="Blin", W="Wlin")
            str_loops_lin = self.generateLoops(save_path_lin,W='Wlin',B='Blin',neuron='neuronlin_of_L',Layer='Layerlin',out  = 'Ilat1_lin')


            # generate log model
            if(self.I_model_pars["log"] != None) :
                print("log model exists")
                if ("mdl_arch" in self.I_model_pars["log"]) and (self.I_model_pars["log"]["mdl_arch"] == "SR"):
                    print("log model is SR")
                    i_mdl_log = feyn.Model.load(self.sr_mdl_dir + "/mx/log/"+ str(self.I_model_pars["log"]["_id"]) )
                    str_weights_log = ""
                    str_model_log =  f"\nsigma_log =  " + str(i_mdl_log.sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)") + ";\n"
                
                else:
                    print("log model is NN")
                    path_log = self.nn_mdl_dir + "log/" + str(self.I_model_pars["log"]["_id"])
                    self.model_log = keras.models.load_model(path_log,custom_objects = { 'R2': NN_u.R2})

                    save_path_log = self.nn_mdl_dir + "../mx_weights/" + "log/" + str(self.I_model_pars["log"]["_id"])+'.hdf5'
                    w_log = Readout.save_weights(save_path = save_path_log,model_path =path_log)
                    Readout.check_if_saved_and_loaded_weights_are_equal(weights_saved=w_log,weights_path=save_path_log)

                    str_weights_log = self.generateWeights(save_path_log)

                    str_model_log = self.generateLoops(save_path_log,out="sigma_log")
                    

            str_model_log = str_model_log + "//--- Backtransfer ---\n" + f"Ilat1_log = volt[0] *(-1)*exp(sigma_log * {log_postscale}) ;\n"


            at_parameters = f"\n // PARAMETERS\n" + str_weights_log + str_weights_lin + str_other_params
            at_i_model = f"\n // I MODEL\n" + str_model_log + str_loops_lin + str_backend



        q_mdl = {}
        # q_mdl["Qbg"] = feyn.Model.load(self.sr_mdl_dir + "/mx/Q/"+ str(self.Q_model_pars["_id"]) + "_Qbg")
        q_mdl["Qfg"] = feyn.Model.load(self.sr_mdl_dir + "/mx/Q/"+ str(self.Q_model_pars["_id"]) + "_Qfg")
        q_mdl["Qtg"] = feyn.Model.load(self.sr_mdl_dir + "/mx/Q/"+ str(self.Q_model_pars["_id"]) + "_Qtg")
        q_mdl["Qlat1"] = feyn.Model.load(self.sr_mdl_dir + "/mx/Q/"+ str(self.Q_model_pars["_id"]) + "_Qlat1")
        q_mdl["Qlat2"] = feyn.Model.load(self.sr_mdl_dir + "/mx/Q/"+ str(self.Q_model_pars["_id"]) + "_Qlat2")
    
        eq_qbg = "0"
        # eq_qbg = str(q_mdl["Qbg"].sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)")
        eq_qtg = str(q_mdl["Qtg"].sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)")
        eq_qfg = str(q_mdl["Qfg"].sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)")
        eq_qlat1 = str(q_mdl["Qlat1"].sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)")
        eq_qlat2 = str(q_mdl["Qlat2"].sympify(signif= significant_digits,include_weights=True)).replace("Vtglat1", "V(b_TG1)").replace("Vfglat1", "V(b_FG1)").replace("Vlat21", "V(b_lat21)")

        q_eqs = f"Qbg = {eq_qbg};\n" + f"Qtg = {eq_qtg};\n" + f"Qfg = {eq_qfg};\n" + f"Qlat1 = {eq_qlat1};\n" +f"Qlat2 = {eq_qlat2};\n"





        
        at_q_model = f"\n // Q MODEL\n" + f"Qscale = {1/Q_scale};\n\n"   + q_eqs
        at_mdl_pars = str(self.I_model_pars)  + "\n\n\n" + str(self.Q_model_pars)

        va_str = self.str_template.replace("@parameters@", at_parameters).replace("@Imodel@", at_i_model).replace("@Qmodel@", at_q_model).replace("@mdl_pars@", at_mdl_pars)


        
        va_file = open(path,"w")
        va_file.write(va_str)
        va_file.close()

        # parameters



    def generateWeights(self, hdf5_path, B = "B", W = "W"):

        LayerList,NameList,WeightBiasList = Readout.load_weights(mypath=hdf5_path)

        str_weights = ""

        
        for n in range(len(LayerList)):
            # --- will give Weights as np.ndarray (Matrix) ---
            curr_weight = WeightBiasList[2*n+1]
            # --- will give Bias as array ---
            curr_bias = WeightBiasList[2*n]
            
            if (np.size(curr_weight) <= np.size(curr_bias)):
                warnings.warn('Weight matrix does not have more elements than bias vec')
            
            n_cols = curr_weight.shape[1]
            n_rows = curr_weight.shape[0]

            str_weights =  str_weights +'// --- '+LayerList[n]+' ---'
            str_weights =  str_weights +'\n'
            # --- writing weights ---
            for i in range (n_rows):
                str_weights =  str_weights +'real  '
                str_weights =  str_weights +W+str(n)+str(i)
                str_weights =  str_weights +'[0:'+str(n_cols-1)+'] = '
                row_arr = curr_weight[i,:]
                row_str = np.array2string(row_arr, separator=', ',floatmode='maxprec',
                formatter={'all':lambda x: "%.9e" % x},max_line_width=700)
                row_str = row_str[1:-1]
                row_str = '{'+row_str+'};'
                str_weights =  str_weights +row_str
                str_weights =  str_weights +'\n'
            # --- writing bias ---
            str_weights =  str_weights +'\n'
            str_weights =  str_weights +'real  '
            str_weights =  str_weights +B+str(n)
            str_weights =  str_weights +'[0:'+str(n_cols-1)+'] = '
            bias_str = np.array2string(curr_bias, separator=', ',floatmode='maxprec',
                formatter={'all':lambda x: "%.9e" % x},max_line_width=700)
            bias_str = bias_str[1:-1]
            bias_str = '{'+bias_str+'};'
            str_weights =  str_weights +bias_str
            str_weights =  str_weights +'\n'

        return str_weights


    def generateLoops(self,hdf5_path,
                    W = 'W' ,
                    B = 'B' ,
                    neuron = 'neuron_of_L_' ,
                    Layer = 'Layer' ,
                    out = 'out'
                    ):
        """ Wirtes the loops responsible for forward progagation of the NN (prediction). 
            Args:
                myfile: path of file to write to
                hdf5_path: path of hdf5 file containing the model. Aquired by running NN_Readout.save_weights
                W: 'Weights prefix'. Change e.g. to Wlin adn Wlog for an ensembled model.
                B: Bias prefix
                neuron: prefix for neurons
                Layer: prefix for Layer
                out: Name for output. Change for example to sigmalog for log model

        """
        LayerList,NameList,WeightBiasList = Readout.load_weights(mypath=hdf5_path)
        
        str_loops = '\n\n// --- loops ---\n'
        
        for n in range(len(LayerList)):
            # --- will give Weights as np.ndarray (Matrix) ---
            curr_weight = WeightBiasList[2*n+1]
            # --- will give Bias as array ---
            curr_bias = WeightBiasList[2*n]
            
            if (np.size(curr_weight) <= np.size(curr_bias)):
                warnings.warn('Weight matrix does not have more elements than bias vec')
            
            n_cols = curr_weight.shape[1]
            n_rows = curr_weight.shape[0]


            str_loops = str_loops + '// --- '+LayerList[n]+' ---\n'
            str_loops = str_loops + 'for (j=0;j<'+str(n_cols)+';j=j+1) begin\n'

            calc_str = ''
            for i in range(n_rows):
                if n == 0:
                    a_string = 'volt['+str(i)+']'
                else:
                    a_string = Layer+str(n-1)+'_out['+str(i)+']'
                calc_str = calc_str + W+str(n)+str(i)+'[j]*'+a_string+' + '

            str_loops = str_loops + neuron+str(n)+' = '+ calc_str
            str_loops = str_loops + B+str(n)+'[j];'
            str_loops = str_loops + '\n'
            if n == len(LayerList)-1:
                str_loops = str_loops + out+' = '+neuron+str(n)+';'
            else:
                str_loops = str_loops + Layer+str(n)+'_out[j] = tanh('+neuron+str(n)+');'
            str_loops = str_loops + '\nend\n'

        return str_loops
    
    def predict(self, X):

        def log10(x):
            x1 = keras.math.log(x)
            x2 = keras.math.log(10.0)
            return x1/ x2

        if (self.I_model_pars["lin"]["mdl_arch"]  == "NN" ) and (self.I_model_pars["log"]["mdl_arch"] == "NN") : 
            y_pred_lin = self.model_lin(X)
            sigma_log = self.model_log(X)

            # inv Thung
            y_pred_log = keras.exp(sigma_log)
            y_pred_log = keras.multiply(y_pred_log, float(X[0][0])) 

            # tanh backend
            epsilon = 1e-100
            slope_scale = self.weight_slope
            offset = self.weight_offset
            k_var = 0.5*log10(keras.square(y_pred_log)+epsilon)
            beta = 0.5*(1+keras.tanh(slope_scale*k_var+offset))

            out = beta* y_pred_lin+ (1-beta)*y_pred_log


        return y_pred_lin, y_pred_log, out


    def sim_cds_inv(self, dir_ocn = "./", dir_results = "./", template_file="../templates/inv_template.ocn" , n_steps = 100):

        # clean results directory
        files  = glob.glob(dir_results + "/*")
        for f in files:
            if os.path.isfile(f ):
                os.remove(f)
            else:
                shutil.rmtree(f)


        # write ocean script

        with open(template_file, 'r') as ocn_file:
            ocn_str = ocn_file.read()

        ocn_str = pyt.dict_replace_parameters(ocn_str, {"@name@"        : str(self.name), 
                                                        "@export_dir@"  : dir_results, 
                                                        "@n_steps@"     : n_steps})

        with open(dir_ocn + "/" + "inv.ocn", 'w') as ocn_file:
            ocn_file.write(ocn_str)


        # simulate (execute ocean script)

        env_vars = dict(os.environ)
        env_vars["LM_LICENSE_FILE"] = "27000@idefix"
        env_vars["CDS_LIC_FILE"] = "5280@idefix"

        env_vars["PATH"] = env_vars["PATH"] + ":" + "/eda/cadence/2021-22/RHELx86/PVS_21.10.000/bin:/eda/cadence/2021-22/RHELx86/QUANTUS_21.11.000/bin:/eda/cadence/2021-22/RHELx86/ASSURA_04.16.111_618/tools/bin:/eda/cadence/2021-22/RHELx86/ASSURA_04.16.111_618/tools/assura/bin:/eda/cadence/2021-22/RHELx86/ASSURA_04.16.111_618/bin:/eda/cadence/2021-22/RHELx86/MVS_21.12.000/bin:/eda/cadence/2021-22/RHELx86/IC_6.1.8.210/tools/bin:/eda/cadence/2021-22/RHELx86/IC_6.1.8.210/tools/dfII/bin:/eda/cadence/2021-22/RHELx86/LIBERATE_21.11.316/bin:/eda/cadence/2021-22/RHELx86/CONFRML_21.10.300/bin:/eda/cadence/2021-22/RHELx86/INNOVUS_21.11.000/bin:/eda/cadence/2021-22/RHELx86/GENUS_21.10.000/tools/bin:/eda/cadence/2021-22/RHELx86/SSV_21.11.000/bin:/eda/cadence/2021-22/RHELx86/SPECTRE_21.10.132/bin:/eda/cadence/2021-22/RHELx86/XCELIUM_21.03.009/bin:/eda/cadence/2021-22/RHELx86/XCELIUM_21.03.009/tools/bin:/eda/cadence/2021-22/RHELx86/XCELIUM_21.03.009/tools/cdsgcc/gcc/bin:/eda/cadence/2021-22/RHELx86/VIPCAT_11.30.079/tools/bin:/eda/cadence/2021-22/RHELx86/JLS_21.10.000/bin:/eda/cadence/2021-22/RHELx86/MODUS_21.10.000/bin:/cad/mentor/mentor/2020-21/RHELx86/"

        pathlib.Path(dir_results + "/" + str(self.name) ).mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(["ocean", "-replay", "inv.ocn"],  env=env_vars, cwd=dir_ocn)
        # self.proc = subprocess.run(["sleep", "200"], env=env_vars, cwd=SIM_TMP_DIR)

        proc.wait()

        df = pd.DataFrame()


        filelist = glob.glob(os.path.join(dir_results+ "/" + str(self.name), "*"))
        sweep_name= "VA"

        for f in filelist:
            if  (".csv" in f):
                #exclude large files (outfiles have ending _des.tdr)
                with open(f, 'rb') as csv_file:
                    dir, file = os.path.split(f)
                    file_str = csv_file.read()

                    df_file = pd.read_csv(dir + "/" + file, skiprows=5, delimiter="\s+", header=None)
                    trace_name = file.split(".")[0]

                    if(trace_name == sweep_name):
                        df_file.columns = [trace_name, trace_name + "_y"]
                    else:
                        df_file.columns = [sweep_name, trace_name]

                    df = pd.concat([df, df_file[trace_name]], axis=1)

        self.trace_cds_inv = df
        # res = cds_sim.get_results(dir_results + "/" + sim_name , "VA")


    @staticmethod
    def getCleanRail(df : pd.DataFrame , Vdd, threshold, area, xname ="VA", yname="VOUT_NN_inv1", plot=False):
        if df[df[xname] < area*Vdd][yname].min() < Vdd-Vdd*threshold:
            return False
        
        if df[df[xname] > Vdd-area*Vdd][yname].max() > Vdd*threshold:
            return False

    @staticmethod
    def compareToRef(df : pd.DataFrame , df_ref : pd.DataFrame, threshold, area, xname ="VA", yname="VOUT_NN_inv1", plot=False):
        pass


    @staticmethod
    def plotINVVTC(df : pd.DataFrame, xname ="VA", yname="VOUT_NN_inv1"):


        fig = plt.figure()
        # trace_cds_inv_tmp = mdl.trace_cds_inv.copy()

        # trace_cds_inv_tmp = pd.concat([trace_cds_inv_tmp, pd.DataFrame({"VA" : [None], "VOUT_NN_inv1" : [y_intersect]})])
        # trace_cds_inv_tmp.sort_values(["VOUT_NN_inv1"], inplace=True)
        # trace_cds_inv_tmp.reset_index(inplace=True)
        # trace_cds_inv_tmp = trace_cds_inv_tmp.interpolate(method="linear")





        plt.plot(df["VA"], df["VOUT_NN_inv1"], ".")

        ax1 = fig.get_axes()[0]

        ax2= ax1.twinx()
        ax2.set_yscale("log")

        ax2.plot(df["VA"], df["p_Ilog"].abs(), ".")
        # ax1.set_yticks(list(ax1.get_yticks()) + [y_high , y_low])

        # ax1.grid()

        return  fig
    
    @staticmethod
    def getTransition(df : pd.DataFrame , yhigh, ylow, xname ="VA", yname="VOUT_NN_inv1", plot=False):

        df_tmp = df.copy()

        df_tmp = pd.concat([df_tmp, pd.DataFrame({xname : [None], yname  : [ylow]})])
        df_tmp = pd.concat([df_tmp, pd.DataFrame({xname : [None], yname : [yhigh]})])
        df_tmp = pd.concat([df_tmp, pd.DataFrame({xname : [None], yname : [(yhigh+ylow)/2]})])

        df_tmp.sort_values([yname], inplace=True)
        df_tmp.reset_index(inplace=True)
        df_tmp = df_tmp.interpolate(method="linear", limit_direction="forward", axis=0)

        xlow = float(df_tmp[df_tmp[yname] == ylow][xname].iloc[0])
        xhigh =  float(df_tmp[df_tmp[yname] == yhigh][xname].iloc[0])

        xcenter =  float(df_tmp[df_tmp[yname] == (yhigh+ylow)/2][xname].iloc[0])

        width = abs(xhigh-xlow)


        fig = None

        if(plot==True):
            fig = plt.figure()

            y_high  = 0.9 * 1.4
            y_low   = 0.1 * 1.4

            print(f" xlow: {xlow}\nxhigh: {xhigh}\nwidth: {width}\nxcenter: {xcenter}")
            # trace_cds_inv_tmp = mdl.trace_cds_inv.copy()

            # trace_cds_inv_tmp = pd.concat([trace_cds_inv_tmp, pd.DataFrame({"VA" : [None], "VOUT_NN_inv1" : [y_intersect]})])
            # trace_cds_inv_tmp.sort_values(["VOUT_NN_inv1"], inplace=True)
            # trace_cds_inv_tmp.reset_index(inplace=True)
            # trace_cds_inv_tmp = trace_cds_inv_tmp.interpolate(method="linear")





            plt.plot(df_tmp[xname], df_tmp[yname], ".")

            ax1 = fig.get_axes()[0]

            ax1.plot([0,1.4], [y_high, y_high], c="gray", linewidth=1, linestyle="--")
            ax1.plot([0,1.4], [y_low, y_low], c="gray", linewidth=1, linestyle="--")

            ax1.scatter([xlow, xhigh], df_tmp[(df_tmp[xname] == xlow) | (df_tmp[xname] == xhigh)][yname])

            rect_ex = patches.Rectangle((xhigh,0 ), width, 1.4, linewidth=2, edgecolor='none', facecolor="#f2d0b6", zorder=-10, clip_on=False)
            ax1.add_patch(rect_ex)

            # ax1.set_yticks(list(ax1.get_yticks()) + [y_high , y_low])

            # ax1.grid()

        return  xlow, xhigh, width , xcenter, df_tmp, fig
    
    @staticmethod
    def plotRef(fig, path_ref = "evaluation/refs/tcad/ml_ref_inv_chainn1112_sys_des.plt" ):
        with open(path_ref , 'r') as plt_file:
            plt_file_lines = plt_file.read()

            # extract all trace names
            trace_names = re.findall(r"\"(.+?)\"", plt_file_lines)

            # extract all values of form 3.90968971685765E-01
            trace_values = re.findall(r"-?\d\.\d{14}E[\+-]\d{2}", plt_file_lines)


            # create traces structure
            traces_ref = {}

            for name in trace_names:
                # take every nth entry an add to list
                tmp = trace_values[trace_names.index(name) :: len(trace_names)]
                traces_ref[name] = [float(i) for i in tmp] 

        fig.get_axes()[0].plot(traces_ref["v(Vin)"], traces_ref["v(Vout_1)"], ".")


    def get_score(self,x_lin_test, y_lin_test, x_raw_test, y_raw_test, prefix=""):
        
        
        ####### LIN
        y_pred = self.model_lin.predict(x_lin_test)

        r2 = skm.r2_score(y_lin_test,y_pred)
        mae = skm.mean_absolute_error(y_lin_test,y_pred)
        mse = skm.mean_squared_error(y_lin_test,y_pred)
        smape = uf.Smape(np.array(y_lin_test),np.array(y_pred).reshape(-1,1))

        score_lin = {f"§{prefix}_Ilin_r2§" :r2, f"§{prefix}_Ilin_mae§" : mae, f"§{prefix}_Ilin_mse§":  mse, f"§{prefix}_Ilin_smape§" : smape}

        ####### LOG

        # y_pred_tmp = self.model_log.predict(x_test)

        # y_pred_mA =- np.multiply(np.array(x_test["Vlat21"]), np.exp(np.array(y_pred_tmp).reshape([1,-1])[0]) )

        test = pd.DataFrame()
        test = pd.concat([x_raw_test,y_raw_test], axis=1)

        r2, mae, mse, smape = uf.evaluate_test_log_all(self.model_log, test, Xlabels=["Vlat21", "Vfglat1", "Vtglat1"], Ylabel=["ids"])

        score_log = {f"§{prefix}_Ilog_r2§" :r2, f"§{prefix}_Ilog_mae§" : mae, f"§{prefix}_Ilog_mse§":  mse, f"§{prefix}_Ilog_smape§" : smape}

        score = score_lin
        score.update(score_log)

        ####### I model (ensemble)

        # y_pred_I = self.predict_nn(x_raw_test)

        def log10(x):
            x1 = np.log(x)
            x2 = np.log(10.0)
            return x1/ x2


        y_pred_lin = self.model_lin.predict(x_raw_test).reshape([1,-1])[0]
        sigma_log = self.model_log.predict(x_raw_test).reshape([1,-1])[0]

        # inv Thung
        y_pred_log = np.exp(sigma_log)
        y_pred_log = - np.multiply( y_pred_log,x_raw_test["Vlat21"].to_numpy().reshape([1,-1])[0])

        # tanh backend
        epsilon = 1e-100
        slope_scale = self.weight_slope
        offset = self.weight_offset
        k_var = 0.5*log10(np.square(y_pred_log)+epsilon)
        beta = 0.5*(1+np.tanh(slope_scale*k_var+offset))

        I_pred = beta* y_pred_lin+ (1-beta)*y_pred_log

        # I_pred = I_pred.reshape([1,-1])[0]
        y_raw_test = y_raw_test.to_numpy().reshape([1,-1])[0]
        r2 = skm.r2_score(y_raw_test,I_pred)
        mae = skm.mean_absolute_error(y_raw_test,I_pred)
        mse = skm.mean_squared_error(y_raw_test,I_pred)
        smape = uf.Smape(y_raw_test.reshape([-1,1]),I_pred.reshape([-1,1]))

        score_I = {f"§{prefix}_I_r2§" :r2, f"§{prefix}_I_mae§" : mae, f"§{prefix}_I_mse§":  mse, f"§{prefix}_I_smape§" : smape}
        score.update(score_I)

        return score
    
        # print(f'R2 current model: %.6f' % skm.r2_score(y_true,y_pred))
        # print('MAE current model: %.4E' %(skm.mean_absolute_error(y_true,y_pred)))
        # print('MSE current model: %.4E' %(skm.mean_squared_error(y_true,y_pred)))
        # print('MAPE current model: %.4E' %(skm.mean_absolute_percentage_error(y_true,y_pred)))
        # print('sMAPE current model: %.6f' %(uf.Smape(y_true,y_pred)))
                