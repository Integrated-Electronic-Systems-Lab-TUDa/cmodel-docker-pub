#author: maxr

import sys



import pandas as pd
import numpy as np
import pickle
import re
#import pytaurus_custom_export_sim as pyc
#import pytaurus_tools

# class compact_model:
#     #list of electrodes, sorted according to table columns
#     electrodes = []


def read_TCAD_plt(file):
        
    with open(file) as f:
        plt_str = f.read()
    
    # plt_str = str( sim_files["sim"+str(sim['_id'])+ '_currentfile.plt'])
    
    # extract all trace names
    trace_names = re.findall(r"\"(.+?)\"", plt_str)

    # extract all values of form 3.90968971685765E-01
    trace_values = re.findall(r"-?\d\.\d{14}E[\+-]\d{2}", plt_str)


    # create traces structure
    traces = {}

    for name in trace_names:
        # take every nth entry an add to list
        tmp = trace_values[trace_names.index(name) :: len(trace_names)]
        traces[name] = [float(i) for i in tmp]

    return pd.DataFrame(traces)


def filter_minstep(df, min_step, col ="time"):

    if min_step > 0.0:
        # #filter out all samples that do not obey minimum step size
        # mask = df[col].diff().fillna(min_step) >= min_step
        # df = traces_used[mask]

        # create mask
        mask = df[col].diff().fillna(min_step) > 0


        while(df[mask][col].diff().fillna(min_step).min() < min_step):
            mask = mask & ~(df.index == df[mask][col].diff().fillna(min_step).idxmin())


        return df[mask]
    else:
        return df
    

def load_ps_trans_data(data_path, Xlabels, Ylabel, n = 1, sort_by = ["ids"],min_step = 0.0, scale_factor=1E3):
    columns = Xlabels + Ylabel 

    scale_factor = (-1)*scale_factor
    # all_data = pd.read_table(data_path)

    all_data = pd.read_pickle(data_path)



    # with open(data_path, 'r') as plt_file:
    #   plt_file_lines = plt_file.read()

    #   # extract all trace names
    #   trace_names = re.findall(r"\"(.+?)\"", plt_file_lines)

    #   # extract all values of form 3.90968971685765E-01
    #   trace_values = re.findall(r"(-?\d\.\d{14}E[\+-]\d{2})", plt_file_lines)


    #   # create traces structure
    #   traces = {}

    #   for name in trace_names:
    #       # take every nth entry an add to list
    #       tmp = trace_values[trace_names.index(name) :: len(trace_names)]
    #       traces[name] = [float(i) for i in tmp]

    # traces["Vlat21"] = traces.pop("lat2 OuterVoltage")
    # traces["Vfglat1"] = traces.pop("fgate OuterVoltage")
    # traces["Vtglat1"] = traces.pop("tgate OuterVoltage")
    # traces["Vbglat1"] = traces.pop("bgate OuterVoltage")

    # traces["Qfg"] = traces.pop("fgate Charge")
    # traces["Qtg"] = traces.pop("tgate Charge")
    # traces["Qbg"] = traces.pop("bgate Charge")
    # traces["Qlat2"] = traces.pop("lat2 Charge")
    # traces["Qlat1"] = traces.pop("lat1 Charge")

    # traces["ids"] = traces.pop("lat2 TotalCurrent")

    all_data["Vfglat1"] = all_data["fgate OuterVoltage"]
    all_data["Vtglat1"] = all_data["tgate OuterVoltage"]
    all_data["Vlat21"] = all_data["lat2 OuterVoltage"]
    all_data["ids"] = all_data["lat2 TotalCurrent"]

    all_data["Qbg"] = all_data["bgate Charge"]
    all_data["Qfg"] = all_data["fgate Charge"]
    all_data["Qtg"] = all_data["tgate Charge"]
    all_data["Qlat2"] = all_data["lat2 Charge"]
    all_data["Qlat1"] = all_data["lat1 Charge"]

    # all_data = filter_minstep(all_data, min_step, col=Xlabels[-1])


    all_data["ids"] = all_data["ids"].apply(lambda x: x*scale_factor)

    data = all_data.iloc[::n,:]

    if (sort_by != None):
        # sort data (important for SR)
        data = data.sort_values(by=sort_by)
        data = data.reset_index(drop=True)

    data = all_data[columns]

    return data


def load_dc_ramp_data(data_path, Xlabels, Ylabel, sort_by = ["ids"], scale_factor=1E03):
    columns = Xlabels + Ylabel

    all_data = pd.read_pickle(data_path)

    data = all_data

    # all_data.columns =["v1","v2","v3","v4",yname]
    if "ids" in data.columns:
        data["ids"] = data["ids"].apply(lambda x: x*scale_factor)

    if (sort_by != None):
        # sort data (important for SR)
        data = data.sort_values(by=sort_by)
        data = data.reset_index(drop=True)

    data = data[columns]

    return data

def load_table(data_path, Xlabels, Ylabel):
    columns = Xlabels + Ylabel

    scale_factor = 1E03
    all_data = pd.read_table(data_path)
    all_data.columns = [ "Vlat21",  "Vfglat1", "Vtglat1", "ids"]

    data = all_data[columns]
    
    # all_data.columns =["v1","v2","v3","v4",yname]
    data[Ylabel[0]] = data[Ylabel[0]].apply(lambda x: x*scale_factor)

    return data

# load all trace signals
def load_dc_ramp_data_full(data_path, Xlabels, Ylabel):
    columns = Xlabels + Ylabel

    scale_factor = 1E03
    data = getSimData("run_optimized_common_goal")

    data["Vbglat1"] = data["bgate OuterVoltage"] -data["lat1 OuterVoltage"]
    data["Vfglat1"] = data["fgate OuterVoltage"] -data["lat1 OuterVoltage"]
    data["Vtglat1"] = data["tgate OuterVoltage"] -data["lat1 OuterVoltage"]
    data["Vlat21"] = data["lat2 OuterVoltage"] -data["lat1 OuterVoltage"]
    data["ids"] = data["lat2 TotalCurrent"]

    data["Qfg"] = data["fgate Charge"]
    data["Qtg"] = data["tgate Charge"]

    data = data[columns]
    
    # all_data.columns =["v1","v2","v3","v4",yname]
    data["ids"] = data["ids"].apply(lambda x: x*scale_factor)


    return data


def load_namlab_table(data_path, Xlabels, Ylabel):

    ren_cols = {
        "Vbg"   : "Vbglat1",
        "Vpg"   : "Vpglat1",
        "Vds"   : "Vlat21",
        "Vgs"   : "Vcglat1",

        "Id"   : "Ilat2"

    }
    
    columns = Xlabels + Ylabel

    # scale_factor = 1E03
    all_data = pd.read_csv(data_path, sep=" ")
    # all_data.columns = [ "Vlat21",  "Vfglat1", "Vtglat1", "ids"]

    all_data.rename(columns=ren_cols, inplace=True)


    data = all_data[columns]
    
    # all_data.columns =["v1","v2","v3","v4",yname]
    # data[Ylabel[0]] = data[Ylabel[0]].apply(lambda x: x*scale_factor)

    return data

def load_namlab_excel(data_path, Xlabels, Ylabel = ["ids"], sort_by = ["ids"]):
    # list of worksheet names
    n_worksheets = ['n,Vcg1.4', 'n,Vcg1.5', 'n,Vcg1.6', 'n,Vcg1.7', 'n, Vcg1.8']
    p_worksheets = ['p, Vcg0.4', 'p, Vcg0.3', 'p, Vcg0.2', 'p, Vcg0.1', 'p, Vcg0']

    # read data out of excel file, sheet_name = None makes it take all worksheets, or list for specific worksheets
    n_data_dict = pd.read_excel(data_path, sheet_name=n_worksheets)
    p_data_dict = pd.read_excel(data_path, sheet_name=p_worksheets)

    # put worksheets together in one dataframe, seems to reduce performance if concat([dfs to be concatinated])
    n_data_list = []
    p_data_list = []

    for worksheet in n_worksheets:
        n_data_list.append(n_data_dict[worksheet])
    for worksheet in p_worksheets:
        p_data_list.append(p_data_dict[worksheet])

    n_data_original = pd.concat(n_data_list, ignore_index = True)
    p_data_original = pd.concat(p_data_list, ignore_index = True)

    # retain only the first 6 columns (only for n_data needed)
    n_data_original = n_data_original[n_data_original.columns[:6]]

    # A = Control Gate, C = Source, G = Drain, E = Program Gate
    header = ['Vcg', 'Ilat1', 'Vlat1', 'Vpg', 'Ilat2', 'Vlat2']
    n_data = n_data_original.copy()
    p_data = p_data_original.copy()

    n_data.columns = header
    p_data.columns = header

    data = pd.concat([p_data[['Vlat2', 'Vlat1', 'Vcg', 'Vpg', 'Ilat1']], n_data[['Vlat2', 'Vlat1', 'Vcg', 'Vpg', 'Ilat1']]], ignore_index=True)

    # # add Vbg = 0 values
    # p_data.insert(3, 'Vbg', np.zeros(p_data['Vd'].size))
    # n_data.insert(3, 'Vbg', np.zeros(n_data['Vd'].size))

    # -1.8V to set Vs at 0 for all datapoints
    # p_data[['Vcg', 'Vs', 'Vpg', 'Vd']] = p_data[['Vcg', 'Vs', 'Vpg', 'Vd']] - p_data['Vs'][0]
    data[['Vcglat1', 'Vlat1', 'Vpglat1', 'Vlat21']] = data[['Vcg', 'Vlat1', 'Vpg', 'Vlat2']].sub(data['Vlat1'], axis=0)


    data["ids"] = data["Ilat1"]

    # renaming like Xlabel und Ylabel
    # Xlabels =["Vlat21", "Vfglat1", "Vtglat1", "Vbglat1"], Ylabel=["ids"])
    # data.columns = ["Vfglat1", "ids", "Vtglat1", "Vlat21", "Vbglat1"]
    # data.columns = ["Vlat21", "Vfglat1", "Vtglat1", "Vbglat1", "ids"]
    # data.columns = ["Vlat21", "Vfglat1", "Vtglat1", "ids"]

    data.sort_values(sort_by, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data = data[Xlabels + Ylabel]

    return data


def load_namlab_excel2(data_path, Xlabels, Ylabel = ["ids"], sort_by = ["ids"], sheet=None):
        
    ren_cols = {
        "AV" : "Vcg",
        "AI" : "Icg",

        "CV" : "Vlat1",
        "CI" : "Ilat1",

        "GV" : "Vlat2",
        "GI" : "Ilat2",

        "EV" : "Vpg",
        "EI" : "Ipg",

    }

    data = pd.DataFrame()

    for file in data_path:
        xlc_file    = pd.read_excel(file , sheet_name=sheet)

        if sheet == None:
            del xlc_file["Calc"]
            del xlc_file["Settings"]

        
            for sheet in xlc_file:
                data = pd.concat([data, xlc_file[sheet]])
        else:
            data = xlc_file
    

    data.rename(columns=ren_cols, inplace=True)

    data[['Vcglat1', 'Vlat1', 'Vpglat1', 'Vlat21']] = data[['Vcg', 'Vlat1', 'Vpg', 'Vlat2']].sub(data['Vlat1'], axis=0)


    if "Ilat1" in data:
        data["ids"] = data["Ilat1"]

    # renaming like Xlabel und Ylabel
    # Xlabels =["Vlat21", "Vfglat1", "Vtglat1", "Vbglat1"], Ylabel=["ids"])
    # data.columns = ["Vfglat1", "ids", "Vtglat1", "Vlat21", "Vbglat1"]
    # data.columns = ["Vlat21", "Vfglat1", "Vtglat1", "Vbglat1", "ids"]
    # data.columns = ["Vlat21", "Vfglat1", "Vtglat1", "ids"]

    data.sort_values(sort_by, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data = data[Xlabels + Ylabel]

    return data


def load_namlab_excel_single(data_path, Xlabels, Ylabel = ["ids"], sort_by = ["ids"], sheet=None):
        
    ren_cols = {
        "AV" : "Vcglat1",
        "AI" : "Icglat1",

        "CV" : "Vlat21",
        "CI" : "Ilat21",

        "GV" : "Vlat21",
        "GI" : "Ilat21",

        "EV" : "Vpglat1",
        "EI" : "Ipglat1",
    }


    data = pd.DataFrame()

    for file in data_path:
        xlc_file = pd.read_excel(file , sheet_name=sheet)

        if sheet == None:
            for sheet in xlc_file:
                data = pd.concat([data, xlc_file[sheet]])
        else:
            data = xlc_file
    

    data.rename(columns=ren_cols, inplace=True)

    # data[['Vcglat1', 'Vlat1', 'Vpglat1', 'Vlat21']] = data[['Vcg', 'Vlat1', 'Vpg', 'Vlat2']].sub(data['Vlat1'], axis=0)


    if "Ilat21" in data:
        data["ids"] = -data["Ilat21"]

    # renaming like Xlabel und Ylabel
    # Xlabels =["Vlat21", "Vfglat1", "Vtglat1", "Vbglat1"], Ylabel=["ids"])
    # data.columns = ["Vfglat1", "ids", "Vtglat1", "Vlat21", "Vbglat1"]
    # data.columns = ["Vlat21", "Vfglat1", "Vtglat1", "Vbglat1", "ids"]
    # data.columns = ["Vlat21", "Vfglat1", "Vtglat1", "ids"]

    data.sort_values(sort_by, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data = data[Xlabels + Ylabel]

    return data



# def getSimData(run):
#     #get simulations

#     col = mdb["simulations"]

#     aggr_match =[ {"$match" : {"run": run , "status" : "post_processed"} }] 
#     aggr_result = col.aggregate(aggr_match)

#     tick_postprocessed = 0

#     traces_all = pd.DataFrame()

#     for sim in aggr_result:
#         # CURRENTFILE
#         #f = open(tmp_folder + '/' + str(sim['_id'])+ '_currentfile.plt', "wb")
#         #f.write(sim['sim_data'][str(sim['_id'])+ '_currentfile.plt'])
#         #f.close()
        
#         sim_files = pyt_FS.getFilesBy_ID(str(sim["_id"]))

#         plt_str = str( sim_files["sim"+str(sim['_id'])+ '_currentfile.plt'])
        
#         # extract all trace names
#         trace_names = re.findall(r"\"(.+?)\"", plt_str)

#         # extract all values of form 3.90968971685765E-01
#         trace_values = re.findall(r"-?\d\.\d{14}E[\+-]\d{2}", plt_str)


#         # create traces structure
#         traces = {}

#         for name in trace_names:
#             # take every nth entry an add to list
#             tmp = trace_values[trace_names.index(name) :: len(trace_names)]
#             traces[name] = [float(i) for i in tmp] 
            
#         if traces_all.empty:
#             traces_all = pd.DataFrame(traces)

#         else:
#             traces_all = pd.concat([traces_all, pd.DataFrame(traces)])


#         pass 
#         # del traces["lat1 DisplacementCurrent"]
#         # del traces["lat2 DisplacementCurrent"]
#         # del traces["fgate DisplacementCurrent"]
#         # del traces["bgate DisplacementCurrent"]
#         # del traces["tgate DisplacementCurrent"]

#         # del traces["lat1 QuasiFermiPotential"]
#         # del traces["lat2 QuasiFermiPotential"]
#         # del traces["fgate QuasiFermiPotential"]
#         # del traces["bgate QuasiFermiPotential"]
#         # del traces["tgate QuasiFermiPotential"]

#         # del traces["lat1 eCurrent"]
#         # del traces["lat2 eCurrent"]
#         # del traces["fgate eCurrent"]
#         # del traces["bgate eCurrent"]
#         # del traces["tgate eCurrent"]

#         # del traces["lat1 hCurrent"]
#         # del traces["lat2 hCurrent"]
#         # del traces["fgate hCurrent"]
#         # del traces["bgate hCurrent"]
#         # del traces["tgate hCurrent"]

#         # del traces["lat1 InnerVoltage"]
#         # del traces["lat2 InnerVoltage"]
#         # del traces["fgate InnerVoltage"]
#         # del traces["bgate InnerVoltage"]
#         # del traces["tgate InnerVoltage"]

#         # add CPU time

#         # log_str = str(sim_files[str(sim['_id'])+ '_outputfile_des.log'])

#         # # extract CPU time
#         # cpu_time = re.findall(r"total cpu:\s(\d+\.\d+)\ss", log_str)
#         # sim['post_processed']['CPU_TIME'] = float(cpu_time[0])
    
#         # # extract Wall Clock time
#         # wallclock_time = re.findall(r"wallclock:\s(\d+\.\d+)\ss", log_str)
#         # sim['post_processed']['WALLCLOCK_TIME'] = float(wallclock_time[0])

#         # host = re.findall(r"Host\sName:\s(\w+)", log_str)
#         # sim['post_processed']['SIM_HOST'] = host[0]

#     return traces_all


# load all trace signals
def load_from_DC_sim(data_path, Xlabels, Ylabel, scale_factor=1E03):
    columns = Xlabels + Ylabel

    # scale_factor = 1E03
    data = read_TCAD_plt(data_path)

    data["Vbglat1"] = data["bgate OuterVoltage"] -data["lat1 OuterVoltage"]
    data["Vfglat1"] = data["fgate OuterVoltage"] -data["lat1 OuterVoltage"]
    data["Vtglat1"] = data["tgate OuterVoltage"] -data["lat1 OuterVoltage"]
    data["Vlat21"] = data["lat2 OuterVoltage"] -data["lat1 OuterVoltage"]
    data["ids"] = data["lat2 TotalCurrent"]

    data["Qfg"] = data["fgate Charge"]
    data["Qtg"] = data["tgate Charge"]

    data = data[columns]
    
    # all_data.columns =["v1","v2","v3","v4",yname]
    data["ids"] = data["ids"].apply(lambda x: x*scale_factor)


    return data




def main(): 
    pass


if __name__ == "__main__":
    main()
