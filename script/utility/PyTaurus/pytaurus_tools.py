import logging as log
import os
import pymongo
import requests
import glob

import pandas as pd
import numpy as np

import itertools

from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator


# logging


class pytaurusFS():
    """Pytaurus filesystem driver"""

    path = ""

    def __init__(self, path):
        self.path = path

    def getFilesBy_ID(self, _id):

        files = {}

        filelist = glob.glob(os.path.join(self.path, str(_id),  "*"))

        for f in filelist:
            if not ( ("_des.tdr" in f) or ("_plotfile.tdr" in f)):
                #exclude large files (outfiles have ending _des.tdr)
                try: 
                    with open(f, 'r') as sim_file:
                        head, tail = os.path.split(f)
                        files[tail] = sim_file.read()
                except:
                    with open(f, 'rb') as sim_file:
                        head, tail = os.path.split(f)
                        files[tail] = sim_file.read()

        return files


    def writeFilesBy_ID(self, _id, files):
        for f in files:
            #write file
            filename = os.path.join(self.path , str(_id) , f)
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            if type(files[f]) is str:
                with open(filename, "w") as file:
                    file.write(files[f])
            else: 
                with open(filename, "wb") as file:
                    file.write(files[f])


def dict_replace_parameters(cmd, pars):

    for p in pars:
        cmd = cmd.replace(p, str(pars[p]))        

    return cmd


def relabel_simulations(mdb, in_field, in_old_value, in_new_value):
        """ mdb_relabel_simulations [field] [old_value] [new_value]
            modifies all simulations where [field]=[old_value] to [field]=[new_value]"""
        
        # MDB_DB = "mdb_tcad_test"

        # # connect and check
        # mclient = pymongo.MongoClient("mongodb://localhost:27017/")
        # if mclient == None:
        #     print("Connection failed")
        #     return -1
        # else:
        #     print("MongoDB connected")

        # # print DBs
        # dblist = mclient.list_database_names()
        # if dblist:
        #     print("\nFound databases: " + str(dblist))
        # # if self.MDB_DB in dblist:
        # #    print("Found database " + self.MDB_DB)

        # print("selecting DB '" + MDB_DB + "'")
        # mdb = mclient[MDB_DB]


        coll = mdb['simulations']

        query = { in_field : in_old_value}

        new_values = {"$set" : { in_field : in_new_value } }

        ret = coll.update_many(query, new_values)

        print(ret.modified_count, " documents updated")

        return ret.modified_count



def get_only_iso_lines(df_table, dict_electrode_voltages):

    # number of dimensions (used for dynamic iso line check)
    n_dim = 3

    del dict_electrode_voltages["bgate"]

    df_table = df_table[df_table["Vbglat1"] == 0]
    df_table.drop(columns=["Vbglat1"], inplace=True)


    # table order from slow to fast, size = n_dim
    table_order = ["Vtglat1", "Vfglat1", "Vlat21"]

    #sort again
    df_table.sort_values(by=[ 'Vtglat1', 'Vfglat1', 'Vlat21'], inplace=True)
    df_table.reset_index(drop=True, inplace=True)


    # drop all duplicates except the ones that are created by tcad.
    # df_table["keep"] = ~(df_table.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1'])) | (df_table.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1']) & (df_table["source"] == "TCAD"))
    # df_table = df_table[~(df_table.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1'])) | (df_table.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1']) & (df_table["source"] == "TCAD"))]
    # df_sims_grid = df_sims_grid[(df_sims_grid.duplicated(subset=['Vfglat1','Vtglat1'])) | (df_sims_grid.duplicated(subset=['Vfglat1','Vtglat1']) & (df_sims_grid["source"] == "TCAD"))]

    # df_table = df_table.interpolate(method='linear', limit_direction='forward', axis=0)

    #drop all duplicates except the first one.
    # df_table = df_table[~(df_table.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1']))]


    for index, row in df_table.iterrows():
        #checks how many duplicates are there for dimension 3,2,1; 2,1; and 1
        dim_duplicate_count = []

        # for dim in range(0,n_dim):
        #     dim_duplicate_count[dim] = df_table

        #dim Vlat21,Vfglat1,Vtglat1
        d2 = {
            table_order[2] : row[table_order[2]],
            table_order[1] : row[table_order[1]],
            table_order[0] : row[table_order[0]]
        }

        cond = []

        mask = df_table[list(d2.keys())].eq(pd.Series(d2)).all(axis=1)
        dim_duplicate_count2 = mask.sum()
        df_table.at[index, "dim_duplicate_count2"] = dim_duplicate_count2
        

        #dim Vlat21,Vfglat1,Vtglat1
        d1 = {
            table_order[1] : row[table_order[1]],
            table_order[0] : row[table_order[0]]
        }

        cond = []

        mask = df_table[list(d1.keys())].eq(pd.Series(d1)).all(axis=1)
        dim_duplicate_count1 = mask.sum()
        df_table.at[index, "dim_duplicate_count1"] = dim_duplicate_count1

                #dim Vlat21,Vfglat1,Vtglat1
        d0 = {
            table_order[0] : row[table_order[0]]
        }

        cond = []

        mask = df_table[list(d0.keys())].eq(pd.Series(d0)).all(axis=1)
        dim_duplicate_count0 = mask.sum()
        df_table.at[index, "dim_duplicate_count0"] = dim_duplicate_count0

        if index == 1573:
            pass


    return df_table




def interpolate_table(df_table, dict_electrode_voltages):

    del dict_electrode_voltages["bgate"]

    df_table = df_table[df_table["Vbglat1"] == 0]
    df_table.drop(columns=["Vbglat1"], inplace=True)

    df_grid = pd.DataFrame(list(itertools.product(*dict_electrode_voltages.values())))

    df_grid.columns = list(dict_electrode_voltages.keys())


    dict_ren = {
        "lat1": "Vlat1",    
        "lat2": "Vlat2",
        "fgate": "Vfg",
        # "bgate": "Vbg",
        "tgate": "Vtg"
    }

    df_grid.rename(columns=dict_ren, inplace=True)


    # reference electrode voltages to lat1
    df_grid["Vfglat1"]  = df_grid["Vfg"]    - df_grid["Vlat1"]
    df_grid["Vtglat1"]  = df_grid["Vtg"]    - df_grid["Vlat1"]
    # df_grid["Vbglat1"]  = df_grid["Vbg"]    - df_grid["Vlat1"]
    df_grid["Vlat21"]   = df_grid["Vlat2"]  - df_grid["Vlat1"]


    df_grid.drop(columns=['Vfg', 'Vtg', 'Vlat2', 'Vlat1'], inplace=True)

    # df_grid = df_grid[df_grid["Vbglat1"] == 0]


    df_grid["source"] = "generated_grid"


    print(df_grid.head())

    df_sims_grid = pd.concat([df_table, df_grid])

    #sort again
    df_sims_grid.sort_values(by=[ 'Vtglat1', 'Vfglat1', 'Vlat21'], inplace=True)

    # drop all duplicates except the ones that are created by tcad.
    df_sims_grid["keep"] = ~(df_sims_grid.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1'])) | (df_sims_grid.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1']) & (df_sims_grid["source"] == "TCAD"))
    df_sims_grid = df_sims_grid[~(df_sims_grid.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1'])) | (df_sims_grid.duplicated(subset=['Vlat21', 'Vfglat1','Vtglat1']) & (df_sims_grid["source"] == "TCAD"))]
    # df_sims_grid = df_sims_grid[(df_sims_grid.duplicated(subset=['Vfglat1','Vtglat1'])) | (df_sims_grid.duplicated(subset=['Vfglat1','Vtglat1']) & (df_sims_grid["source"] == "TCAD"))]

    df_sims_grid_interp = df_sims_grid.interpolate(method='linear', limit_direction='forward', axis=0)



    return df_sims_grid_interp


def interpolate_grid(df_table, dict_electrode_voltages):

    # del dict_electrode_voltages["bgate"]
    
    df_grid = pd.DataFrame(list(itertools.product(*dict_electrode_voltages.values())))

    df_grid.columns = list(dict_electrode_voltages.keys())

 


    dict_ren = {
        "lat1": "Vlat1",    
        "lat2": "Vlat2",
        "fgate": "Vfg",
        "bgate": "Vbg",
        "tgate": "Vtg"
    }

    
    df_grid.rename(columns=dict_ren, inplace=True)

   # reference electrode voltages to lat1
    df_grid["Vfglat1"]  = df_grid["Vfg"]    - df_grid["Vlat1"]
    df_grid["Vtglat1"]  = df_grid["Vtg"]    - df_grid["Vlat1"]
    df_grid["Vbglat1"]  = df_grid["Vbg"]    - df_grid["Vlat1"]
    df_grid["Vlat21"]   = df_grid["Vlat2"]  - df_grid["Vlat1"]


    df_grid.drop(columns=['Vfg', 'Vbg', 'Vtg', 'Vlat2', 'Vlat1'], inplace=True)

    df_grid = df_grid[df_grid["Vbglat1"] == 0]
    df_table = df_table[df_table["Vbglat1"] == 0]


    df_grid["source"] = "generated_grid"
    df_grid = df_grid[['Vlat21', 'Vfglat1', 'Vtglat1', 'source']]

    df_grid.sort_values(by=['Vlat21', 'Vfglat1', 'Vtglat1'], inplace=True)
    df_grid.reset_index(drop=True, inplace=True)

    # df_sims_and_grid = pd.concat([df_table, df_grid])

    # get only values that are on the grid
    # df_grid_only = df_grid.merge(df_table, on=['Vlat2', 'Vfg', 'Vtg', 'Vbg'], how="left")

    # print(len(df_grid_only[df_grid_only["source"] == "TCAD"])/len(df_grid_only))
    # # append the whole grid
    # df_grid_with_values = pd.concat([df_grid, df_grid_only])

    # # sort to have 'source' in order
    # df_grid_with_values.sort_values(by=['Vlat21', 'Vfg', 'Vtg', 'Vbg', 'source'], inplace=True)


    # df_grid_with_values = df_grid_with_values[~(df_grid_with_values.duplicated(subset=['Vlat21', 'Vfg','Vtg','Vbg'])) | (df_grid_with_values.duplicated(subset=['Vlat21', 'Vfg','Vtg','Vbg']) & (df_grid_with_values["source"] == "TCAD"))]

    #sort again
    # df_sims_and_grid.sort_values(by=['Vlat21', 'Vfglat1', 'Vtglat1',  'source'], inplace=True)

    # drop all duplicates except the ones that are created by tcad.
    # df_sims_grid = df_sims_grid[~(df_sims_grid.duplicated(subset=['Vlat21', 'Vfg','Vtg','Vbg'])) | (df_sims_grid.duplicated(subset=['Vlat21', 'Vfg','Vtg','Vbg']) & (df_sims_grid["source"] == "TCAD"))]

    # df_sims_grid_interp = df_sims_and_grid.interpolate(method='linear', limit_direction='forward', axis=0)

    # points = df_table['Vlat21', 'Vfglat1', 'Vtglat1', 'Vbglat1']
    # values = df_table['ids']
    # request = np.array([0,1.6,1.1,1.1])


    # # res = griddata(points, values, request)
    
    grid_x, grid_y, grid_z = np.meshgrid(df_grid['Vlat21'].unique(), df_grid['Vfglat1'].unique(), df_grid['Vtglat1'].unique(),  indexing='ij')

    print(f"unique voltages : {df_grid['Vlat21'].unique()}")

    points = np.array(list(zip(df_table['Vlat21'], df_table['Vfglat1'], df_table['Vtglat1'])))

    values = df_table['ids']

    interpolated_ids = griddata(points, values, (grid_x, grid_y, grid_z), method='linear')

    interpolated_df = pd.DataFrame({
        'Vlat21'    : grid_x.flatten(),
        'Vfglat1'   : grid_y.flatten(),
        'Vtglat1'   : grid_z.flatten(),
        'interpolated_ids': interpolated_ids.flatten()
    })

    telegram_send("fertig :)")

    print(f"found {interpolated_df['interpolated_ids'].isna().sum()} NaN rows")

    # fill NaN with large value to conserve the grid
    interpolated_df.fillna(9999999999, inplace=True)

    # df_sims_grid_interp = df_sims_grid_interp[df_sims_grid_interp["source"] == "generated_grid"]

    return interpolated_df


def findSimForDataPoint(data):
    
    pass


def telegram_send(text):
    #p = os.popen("/bin/bash -i -c telegram_send \"" + str(text) + "\"" )

    params = {"chat_id":"606111847", "text": str(text)}
    url = f"https://api.telegram.org/bot657569180:AAGReZ4rrp_VIynzQje4xgtY0kYgZpl8YNU/sendMessage"
    message = requests.post(url, params=params)
