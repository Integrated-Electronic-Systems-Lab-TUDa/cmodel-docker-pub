import feyn
import pandas as pd
import numpy as np
import sklearn.metrics as skm 
from sklearn.model_selection import train_test_split

def run_manual_ql(random_seed = 9, save_path = './Qlattice/test.json',num_epochs = 3,
starting_model = None,train_size = 0.25,scale = True):
    """build a qlattice, sample it, train the smapled models, prune models, 
    update qlatiice, resmaple ...
    """
 #--- own hyperparams
    scale_factor = 1E3
    #--- not realy a custom hyperparm; defines how many model are kept after each epoch -> more on feyn aka. Qlattice userguides / website
    keep_n = 2000
    random_seed = random_seed
    y_name = 'id'
    train_size = train_size
 
 #--- Parameters
    ql = feyn.QLattice(random_seed)
    output_name = y_name
    kind = "regression"
    stypes = {}
    n_epochs = num_epochs
    threads = 'auto'
    max_complexity = 150
    query_string = None
    loss_function = 'squared_error'
    criterion = None
    sample_weights = None
    function_names = None
    starting_models = None


    all_data = pd.read_table("./data/id.tbl")
    all_data.columns =["v1","v2","v3","v4",y_name]


    #--- possiblility of scaling 
    if scale == True:
        all_data[y_name] = all_data[y_name].apply(lambda x: x*scale_factor)
    print(all_data.describe())
    train,test  = train_test_split(all_data, train_size=train_size,random_state=random_seed)


    #--- auto_run code expansion:
    from time import time

    feyn.validate_data(all_data, kind, output_name, stypes)

    if n_epochs <= 0:
        raise ValueError("n_epochs must be 1 or higher.")

    if threads == "auto":
        threads = feyn.tools.infer_available_threads()
    elif isinstance(threads, str):
        raise ValueError("threads must be a number, or string 'auto'.")

    models = []
    if starting_models is not None:
        models = [m.copy() for m in starting_models]
    m_count = len(models)

    priors = feyn.tools.estimate_priors(all_data, output_name)
    ql.update_priors(priors)

    try:
        start = time()
        for epoch in range(1, n_epochs + 1):
            new_sample = ql.sample_models(
                train,
                output_name,
                kind,
                stypes,
                max_complexity,
                query_string,
                function_names,
            )
            models += new_sample
            m_count += len(new_sample)

            models = feyn.fit_models(
                models,
                data= train,
                loss_function=loss_function,
                criterion=criterion,
                n_samples=None,
                sample_weights=sample_weights,
                threads=threads,
            )
            models = feyn.prune_models(models,keep_n=keep_n)
            elapsed = time() - start

            if len(models) > 0:
                print(feyn.tools.get_progress_label(epoch, n_epochs, elapsed, m_count))
                if epoch % 20 == 0:
                    y_pred_a = models[0].predict(all_data[['v1','v2','v3','v4']])
                    y_true_a = all_data[y_name].to_numpy().reshape(-1,1)
                    print('R2 of current best :'+str(skm.r2_score(y_true_a,y_pred_a)))

            ql.update(models)

        best_models = feyn.get_diverse_models(models)

    except KeyboardInterrupt:
        best_models = feyn.get_diverse_models(models)



    
    for i in range (0,5):
        a = i+20
        lsave_path = './Qlattice/models/mult_id_runs_0'+str(a)+'.json'
        best_models[i].save(lsave_path)
        y_pred_a = best_models[i].predict(all_data[['v1','v2','v3','v4']])
        y_true_a = all_data[y_name].to_numpy().reshape(-1,1)
        print('qlat R2 all :'+str(skm.r2_score(y_true_a,y_pred_a)))