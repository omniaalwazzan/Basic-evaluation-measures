# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:56:09 2023

@author: Omnia
"""
# First, save the ground truth and the prediction when testing the model to a dataframe 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
path = r"D:\BMVC_rebuttel\TruePred_data_op_oa.csv"
data = pd.read_csv(path)
path2 = r"D:\BMVC_rebuttel\TruePred_data_op_oa_os.csv"
data2 = pd.read_csv(path2)

path3 =  r"D:\BMVC_rebuttel\TruePred_data_CNN.csv"
data3 = pd.read_csv(path3)

fpr1, tpr1, thresholds1 = roc_curve(data['true'], data['pred'])
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, thresholds2 = roc_curve(data2['true'], data2['pred'])
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, thresholds3 = roc_curve(data3['true'], data3['pred'])
roc_auc3 = auc(fpr3, tpr3)


 # Plot the first AUC curve
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'CNN-2 branch(AUC = {roc_auc1:.2f})')

# Plot the second AUC curve
plt.plot(fpr2, tpr2, color='navy', lw=2, label=f'CNN-3 branch(AUC = {roc_auc2:.2f})')

plt.plot(fpr3, tpr3, color='red', lw=2, label=f'CNN branch(AUC = {roc_auc2:.2f})')

# Add labels and legends
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

plt.show()





