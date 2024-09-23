import pymongo
from datetime import datetime
import urllib.parse
from pymongo.errors import ConnectionFailure



mclient = None
mdb = None

def mdb_connect(MDB_DB, user, pw):
    # connect and check
    username = urllib.parse.quote_plus(user)
    password = urllib.parse.quote_plus(pw)

    try:

        mclient = pymongo.MongoClient('mongodb://%s:%s@localhost:27017' % (username, password))

        mclient.admin.command('ping')
    except:

        print("connection error")

    mdb = mclient[MDB_DB]

    
    return mdb


def add_simulation(mdb, collection, name, params,):

    #print(parameters)
    sim = {     'name'              : name,
                'run'               : 'benjamin_characterization',
                'data_origin'       : '',
                'comment'           : '',
                'origin'            : 'custom (mongodb_driver.py)',
                'timestamp'         : datetime.now(),
                'pars'        : {}
            }
    

    sim['sim_data']['cmd'] = pyt.dict_replace_parameters(str(cmd_template), parameters)

    sim['sim_data']['par_file'] = parfile
    sim['sim_data']['mesh_file'] = meshfile

    sim_col = mdb["simulations"]
    ret = sim_col.insert_one(sim)
