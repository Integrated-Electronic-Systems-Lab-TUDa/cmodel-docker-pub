import numpy as np
import pandas as pd
import utility_func as uf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import visualize_results


def main():
    augment_train_with_noise(Keep_original=True)
    
    return

def augment_train_with_noise(trainVal: pd.DataFrame = pd.DataFrame(), num_copies = 3,yname = 'ilat',sigma = 1E-14,Keep_original = True):
    """ Augments data with noise. 
        Args:
            trainVal: input Data
            sigma: sigma of gausian noise distribution. Determines the order of magintude of noise added. 
            num_copies: Should additional copies of the input data be created ? if yes: num copies > 1 -> specifies the amout of copies created
            Keep_original: Determines if the dataset without augmentation is part of the output.
        Returns:
            aug_trainVal: The Dataframe containing the augmented (gaus noise adittion) data. 
                        If num copies > 1 ->  Dataframe wil have n times the size of input data. Each time a different noise vector is added. (drawn from gaus distribution)

    """
    # --- we are making a total of 3 copies. to each of our 3 copies we add some noise with a small sigma
    # --- seperate test set if not allready done
    if trainVal.empty:
        trainVal,_ = uf.split_trainVal_test_set(yname='ilat')
    

    #print(trainVal.describe())

    mu  = 0.0
    if Keep_original == False:
        num_copies = num_copies+1

    aug_trainVal = trainVal
    # --- starting from 1 because 0. copy will be original data
    for n in range(1,num_copies):
        ilat_arr = trainVal.loc[:,yname].to_numpy()
        n_values = len(ilat_arr)
       # print('y has a length of: '+ str(n_values))
       # print(ilat_arr.shape)
        noise_samples = np.random.normal(mu,sigma,size = n_values)
       # print(noise_samples.shape)

        df_copy_noise = trainVal.copy(deep=True)
       
        
        df_copy_noise[yname] = np.reshape(np.add(ilat_arr,noise_samples),newshape = (-1,1))
        if (Keep_original == False) & (n == 1):
            aug_trainVal = df_copy_noise

            
        else: 
            aug_trainVal = pd.concat([aug_trainVal, df_copy_noise], ignore_index=True, sort=False, axis=0, join='inner')

    visualize_results.plot_historgramm(aug_trainVal,vol_to_plt=yname,apply_log=True)
    print(aug_trainVal.describe())

    
   
    return aug_trainVal

if __name__ == "__main__":
    main()
