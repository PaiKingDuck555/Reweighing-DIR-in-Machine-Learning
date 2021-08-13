import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import CompasDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score 

def main(): 
    MYprivileged_groups = [{'race': 1}]
    MYunprivileged_groups = [{'race': 0}]
    dataset_compas = load_preproc_data_compas(['race'])
    print(type(dataset_compas)) 
#     dataset_compas_train = dataset_compas.split([0.7], shuffle=True)
#     print(type(dataset_compas_train))
#     dataset_compas_test = dataset_compas.split([0.5], shuffle=True)

    Reweight = Reweighing(unprivileged_groups = MYunprivileged_groups , privileged_groups = MYprivileged_groups)
    Reweight.fit(dataset_compas) 
    new_dataset_compas = Reweight.transform(dataset_compas) 

    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(new_dataset_compas.features)
    Y_train = new_dataset_compas.labels.ravel()

    lmod = LogisticRegression(solver='lbfgs', max_iter=1000)
    lmod.fit(X_train, Y_train, 
            sample_weight=new_dataset_compas.instance_weights)

    Y_pred = lmod.predict(X_train) 
    Y_true = dataset_compas.labels.ravel()
    F1_score = f1_score(Y_pred, Y_true)
    print(F1_score)
main()