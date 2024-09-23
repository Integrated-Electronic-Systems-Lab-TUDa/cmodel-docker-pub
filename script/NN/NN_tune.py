from tensorflow import keras 
import NN_run
import NN_build_small
import keras_tuner
import pandas as pd
import utility_func as uf
from sklearn.model_selection import train_test_split
import NN_run
def main():
    setup_and_run_tuner()
    return
# --- subclasses HyperModel and overwrite the build and fit fuctions
class MyHyperModel(keras_tuner.HyperModel):

    def build(self,hp):
        units = hp.Int("units", min_value=8, max_value=24, step=4)
        #hp.Choice("activation", ["relu", "tanh"])
        activation = 'tanh'
        #dropout = hp.Boolean("dropout")
        lr = hp.Float("lr", min_value=5e-4, max_value=5e-3, sampling="linear")

        #--- call existing model-building code with the hyperparameter values.
        compiled_model = NN_build_small.build_tuning_model(n_neurons=units,lr=lr,act_func=activation)
        return compiled_model

    def fit(self, hp, model,callbacks_tensorboard =None, *args, **kwargs):
         # Assign the model to the callbacks.
        

        return NN_run.run_NN_for_tuning(
            model=model,
            mdl_type='lin',
            epochs=400,
            subtype_num=hp.Int('subtype',min_value=3, max_value = 10, step = 1),
            callback_tensorboard=callbacks_tensorboard,
            delta_early_stop=3e-07
        )


def setup_and_run_tuner():

    # --- look at guide https://keras.io/guides/keras_tuner/getting_started/
    
    my_tuner  = keras_tuner.RandomSearch(
    hypermodel=MyHyperModel(),
    objective="val_loss",
    max_trials=30,
    executions_per_trial=2,
    overwrite=True,
    
    directory="./NN_tuning_res",
    project_name="early_stopping_tuning" ,
)
    tuner = my_tuner
    # --- the method where all the action happens (actual search/ training of all the models)
    tuner.search(callbacks_tensorboard =[keras.callbacks.TensorBoard("./NN_tuning_res/tune07_logs")])
    
    
    # --- everthing below -> Evaluation of search results ---    
    print('--- Summary of tuning ----')
    print(tuner.search_space_summary())
 
   
    n_trials = 100
    trials = tuner.oracle.get_best_trials(n_trials)
    trial_performace = list()
   
    for t in trials:
        performace = list(t.hyperparameters.values.values())
        performace.append(t.score)
        df_of_trial = pd.DataFrame(performace)
        trial_performace.append(df_of_trial)

    df_results = pd.concat(objs=trial_performace,keys=range(len(trial_performace)),
    axis=1,ignore_index=True)
    df_results = df_results.transpose()
    df_results.columns = [*t.hyperparameters.values.keys(),'score']

    print('--- results for best %d trials ---' %n_trials)
    print(df_results.head(n_trials))
    df_results.to_csv("./NN_tuning_res/summary07.csv",index = False)
    print('--- Result summary--- ')
   
    print(tuner.results_summary())

if __name__ == "__main__":
    main()



