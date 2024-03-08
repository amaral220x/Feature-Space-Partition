from fsp import Feature_Space_Partition
from pdc import Probabilistic_Draw_Classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from scipy.io import loadmat
import numpy as np
import os
import time
import sys

def ravel(vec):
    vec = np.ravel(vec)
    vec = vec.astype(int)
    return vec

def worker(data, opt, cv, j):
    # Split the data into train and test set
    X = data['X']
    y = data['y']
    train_index, test_index = list(cv.split(data['X'], data['y']))[j]
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    #X_train, X_test = ravel(X_train), ravel(X_test)
    y_train, y_test = ravel(y_train), ravel(y_test)
    y_test = y_test - 1

    #mdl fsp
    fields_to_remove = {'PD_case', 'NumberOfRuns', 'KFold', 'knn_NumNeighbors'}
    fsp_opt = {k: opt[k] for k in opt if k not in fields_to_remove}

    #Partition the feature space
    rmH_list, _ = Feature_Space_Partition(X_train, y_train, fsp_opt)

    #Classify the test set
    if opt['PD_case'] == 1:
        ypred, prop_value = Probabilistic_Draw_Classifier(X_test, rmH_list, X_train, y_train)
    elif opt['PD_case'] == 2:
        ypred, prop_value = Probabilistic_Draw_Classifier(X_test, rmH_list)
    else:
        raise ValueError('Invalid PD_case value')
    
    #Calculate accuracy
    yP = ravel(ypred)
    yT = ravel(y_test)
        
    acc = 1 - np.mean(yP != yT)

    mask_homo = prop_value >= (1 - opt['p_parameter'])
    mask_homo = mask_homo.flatten()
    mask_hete = ~mask_homo
    
    ytest_homo = ravel(y_test[mask_homo])
    ytest_hete = ravel(y_test[mask_hete])
    ypred_homo = ravel(ypred[mask_homo])
    ypred_hete = ravel(ypred[mask_hete])

    acchomo = 1 - np.mean(ytest_homo != ypred_homo)
    acchete = 1 - np.mean(ytest_hete != ypred_hete)


    return acc, acchomo, acchete
    



def testing(dataset, opt=None):
    # Load the dataset
    data_dir = 'Datasets/'
    data = loadmat(os.path.join(data_dir, dataset))
    defaultOpt = {
        'initial_k': 1,
        'p_parameter': 0.05,
        'h_threshold': 0,
        'dm_case': 2,
        'dm_threshold': 0.5,
        'update_dm_threshold': True,
        'return_full_dm': False,
        'return_full_history': False,
        'iteration_threshold': 2e6,
        'rng_default': False,
        'PD_case': 2,
        'NumberOfRuns': 10,
        'KFold': 10,
        'knn_NumNeighbors': int(np.floor(np.sqrt(len(data['y']))))
    }

    if opt is None:
        opt = defaultOpt
    else:
        defaultOpt.update(opt)
        opt = defaultOpt

    fsp_results = {'accuracy': [], 'accuracy_homo': [], 'accuracy_hete': [], 'runtime': []}

    for i in range(opt['NumberOfRuns']):

        #Creating the Cross-Validation obj
        if opt['KFold'] == len(data['y']):
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=opt['KFold'], shuffle=True)

        #Starting the results matrix
        acc = np.zeros((1, opt['KFold'],))
        acchomo = np.zeros((1, opt['KFold'],))
        acchete = np.zeros((1, opt['KFold'],))
        
        tStart = time.time()
        for j in range(opt['KFold']):
            results = worker(data, opt, cv, j) #Calling FSP and measuring accuracy
            acc[0, j], acchomo[0, j], acchete[0, j] = results

        runtime = time.time() - tStart

        fsp_results['accuracy'].append(np.mean(acc))
        fsp_results['accuracy_homo'].append(np.nanmean(acchomo))
        fsp_results['accuracy_hete'].append(np.nanmean(acchete))
        fsp_results['runtime'].append(runtime)

    output = {'accuracy': [], 'accuracy_homo': [], 'accuracy_hete': [], 'runtime': []}
    output['accuracy'] =  [100 * np.mean(fsp_results['accuracy']), 100 * np.std(fsp_results['accuracy'])]
    output['accuracy_homo'] =  [100 * np.nanmean(fsp_results['accuracy_homo']), 100 * np.nanstd(fsp_results['accuracy_homo'])]
    output['accuracy_hete'] =  [100 * np.nanmean(fsp_results['accuracy_hete']), 100 * np.nanstd(fsp_results['accuracy_hete'])]
    output['runtime'] =  np.sum(fsp_results['runtime'])

    return output


if __name__ == "__main__":
    dataset = sys.argv[1]
    output = testing(dataset)
    print(output)




