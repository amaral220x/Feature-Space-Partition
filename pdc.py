import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import numpy.matlib

def Probabilistic_Draw_Classifier(XTest, rmH, XTrain=None, yTrain=None):
    if XTrain is not None and yTrain is not None:
        # Cluster centroid locations generated by the "Feature_Space_Partition" function
        C_list = []
        for i in range(len(rmH)):
            rmC = rmH[i]['rmC']
            if isinstance(rmC, list):
                rmC = rmC[0]
            C_list.append(np.array(rmC))

        Centroid = np.vstack(C_list)

        # Number of clusters
        nC = Centroid.shape[0]
        nCL = np.max(np.unique(yTrain))
        labels = np.unique(yTrain)

        # Initialize the variables
        ClassLabelNum = np.zeros((nC, nCL))
        ClassLabelProportion = np.zeros((nC, nCL))

        DominantClassLabelProportion = np.zeros(nC)
        DominantClassLabel = np.zeros(nC)

        vecNc = np.zeros(nC)

        # Calculate the euclidean distance between XTrain and Centroid
        Dist = distance.cdist(XTrain, Centroid)
        # Assign each observation in XTrain to its respective cluster
        indice = np.argmin(Dist, axis=1)
        unique, counts = np.unique(indice, return_counts=True)

        for c in range(nC):
            # Class labels in cluster c
            yc = yTrain[indice == c]
            # Number of observations in cluster c
            Nc = len(yc)
            vecNc[c] = Nc
            # For each class label, do:
            for l in range(nCL):
                label = labels[l]
                # Number of observations in cluster c with class label l
                ClassLabelNum[c, l] = np.sum(yc == label)
                # Proportion of the class label l into cluster c
                ClassLabelProportion[c, l] = ClassLabelNum[c, l] / Nc
            #print(f'ClassLabelNum: {ClassLabelNum}')
            #print(f'ClassLabelProportion: {ClassLabelProportion}')
            pc, lc = np.max(ClassLabelProportion[c, :]), np.argmax(ClassLabelProportion[c, :])
            DominantClassLabelProportion[c] = pc
            DominantClassLabel[c] = lc

        # Calculate the euclidean distance between XTest and Centroid
        Dist = distance.cdist(XTest, Centroid)
        # Assign each observation in XTest to its respective cluster
        indice = np.argmin(Dist, axis=1)
        # Assign the i-th entry of yTestPD as the majority class label of the cluster to which the i-th observation XTest(i,:) belongs.
        yTestPD = DominantClassLabel[indice]
        yTestPD_Proportion = DominantClassLabelProportion[indice]
        yTestPD = yTestPD.astype(int)

    elif XTrain is None and yTrain is None:
        # Take a copy of XTest
        X = XTest.copy()

        C_list = []
        for i in range(len(rmH)):
            rmC = rmH[i]['rmC']
            if isinstance(rmC, list):
                rmC = rmC[0]
            C_list.append(np.array(rmC))

        Centroid = np.vstack(C_list)
        #print(f'Centroid: {Centroid}')
        nC = Centroid.shape[0]  # Number of clusters

        # Initialize cell arrays to store cluster-specific data
        cell_X, cell_l, cell_p = [None] * nC, [None] * nC, [None] * nC
        count = 0
        # Iterate through rmH iterations
        for i in range(len(rmH)):
            # Classify the X dataset by finding the nearest centroid for each data point in X during the current iteration
            # idx_test is a row vector with the same number of entries as observations in X.
            # Each element of idx_test represents the cluster assignment for the corresponding data point in X.
            rmHs = rmH[i]['C']
            if isinstance(rmHs, list):
                rmHs = rmHs[0]
            rmHs = np.array(rmHs)

            Dist = distance.cdist(X, rmHs, metric='euclidean')
            idx_test = np.argmin(Dist, axis=1)
            rmHrmC = np.array(rmH[i]['rmCidx'])
            rmHrmC = np.ravel(rmHrmC)
            rdominantClassLabel = np.array(rmH[i]['rmDominantClassLabel'])
            rdominantClassLabel = np.ravel(rdominantClassLabel)
            rdominantClassLabel_Proportion = np.array(rmH[i]['rmDominantClassLabel_Proportion'])
            rdominantClassLabel_Proportion = np.ravel(rdominantClassLabel_Proportion)
            #Find the indices of the observations in X that are in clusters removed in this iteration
            rmidx = []

            for j in range(len(rmHrmC)):
                idx_in_removed_cluster = np.where(idx_test == rmHrmC[j])[0]
                NumberOfObs = len(idx_in_removed_cluster)
                if NumberOfObs > 0:
                    cell_X[count] = X[idx_in_removed_cluster, :]
                    cell_l[count] = np.matlib.repmat(rdominantClassLabel[j], NumberOfObs, 1)
                    cell_p[count] = np.matlib.repmat(rdominantClassLabel_Proportion[j], NumberOfObs, 1)
                    # Keep track of indices for later removal from X
                    rmidx.extend(idx_in_removed_cluster.tolist())
                count += 1
            
            X = np.delete(X, rmidx, axis=0)
        # Check if all observations in X have been assigned to a cluster in 'Centroid.' Raise an error if not.
        if X.shape[0] != 0:
            raise ValueError('An error has occurred in step 1.2. X must be empty.')

        # Concatenate the contents of cell_X in X
        cell_X_list = []
        for i in range(len(cell_X)):
            if cell_X[i] is not None:
                cell_X_list.append(cell_X[i])
        

        X = np.vstack(cell_X_list)
        cell_l_list = []

        for i in range(len(cell_l)):
            if cell_l[i] is not None:
                cell_l_list.append(cell_l[i])

        cell_p_list = []
        for i in range(len(cell_p)):
            if cell_p[i] is not None:
                cell_p_list.append(cell_p[i])

        yPD = np.vstack(cell_l_list)
        yPD_Proportion = np.vstack(cell_p_list)

        # Restore the original order
        neig = NearestNeighbors(n_neighbors=1)
        neig.fit(X)
        idx = neig.kneighbors(XTest, return_distance=False)
        yTestPD = yPD[idx]
        yTestPD_Proportion = yPD_Proportion[idx]

    else:
        raise ValueError('This function is prepared to be used with 2 or 4 input parameters only.')

    return yTestPD, yTestPD_Proportion