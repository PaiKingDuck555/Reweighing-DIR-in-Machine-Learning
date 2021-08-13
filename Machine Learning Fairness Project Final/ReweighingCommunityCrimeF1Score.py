import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score



def main():

    df = pd.read_csv('crimerate.csv', header=None, names=[str(i) for i in range(101)])
    # # binarize protected attribute, african american community
    df.iloc[:,2] = (df.iloc[:,2] <= 0.5)
    # # binarize label. high value (high crimerate) == bad
    df.iloc[:,100] = (df.iloc[:,100] <= 0.5)

    crime_dataset = BinaryLabelDataset(
        df=df, 
        favorable_label=0,
        unfavorable_label=1,
        label_names=['100'], # the last column "label" "crimerate"
        protected_attribute_names=['2'], # ["race_pct_african_american"] 
    )
    underprivileged = [{'2': 0}]
    privileged = [{'2': 1}]
    
    crime_dataset_train = crime_dataset.split([0.7], shuffle=True)
    crime_dataset_test = crime_dataset.split([0.5], shuffle=True)


    var = Reweighing(underprivileged, privileged)
    var.fit(crime_dataset)
    new_dataset = var.transform(crime_dataset) # This is a BinaryLabelDataset
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(new_dataset.features)
    Y_train = crime_dataset_test.labels.ravel()
    new_dataset.instance_weights.ravel()
    lmod = LogisticRegression(solver='lbfgs', max_iter=1000)
    lmod.fit(X_train, Y_train, 
        sample_weight=new_dataset.instance_weights)
    Y_pred = lmod.predict(X_train) 
    Y_true = Y_train 
    F1_score = f1_score(Y_pred, Y_true)
    print(F1_score)
 

## do whatever you want now with the data! Compute some metrics!
       
main()