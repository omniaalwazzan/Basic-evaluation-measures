# Compute the area under the curve of each class as one-vs-all then then avg them

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


truelabels=    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 0, 2, 1, 0, 2]
predictions= [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 2, 1, 1, 2]


conf_matrix = confusion_matrix(truelabels, predictions)

print(classification_report(truelabels, predictions))


    
def confusion_metrics_1(conf_matrix):
    num_classes = conf_matrix.shape[0]
    metrics = {}
    result_str = ""
    
    for i in range(num_classes):
        tp = conf_matrix[i,i]
        fp = np.sum(conf_matrix[:,i]) - tp
        fn = np.sum(conf_matrix[i,:]) - tp
        tn = np.sum(conf_matrix) - tp - fp - fn
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        metrics[f'Class {i}'] = {'Accuracy': accuracy,
                                  'Sensitivity': sensitivity,
                                  'Specificity': specificity}
        result_str += f"Class {i}:\nAccuracy: {accuracy:.2f}\nSensitivity: {sensitivity:.2f}\nSpecificity: {specificity:.2f}\n"

    return result_str

Specificity = confusion_metrics_1(conf_matrix)
print(Specificity)
from sklearn.preprocessing import OneHotEncoder

def one_hot(truelabels):
    # Convert the list to a numpy array and reshape it to a column vector
    target_array = np.array(truelabels).reshape(-1, 1)
    # Initialize the OneHotEncoder object
    encoder = OneHotEncoder()
    # Fit the encoder to the target array
    encoder.fit(target_array)
    # Transform the target array to one-hot encoded format
    one_hot_target = encoder.transform(target_array).toarray()
    #print(one_hot_target)

    return one_hot_target

truelabels = one_hot(truelabels)
predictions = one_hot(predictions)




def multiclass_roc_auc_score(y_true, y_pred, average="macro"):
    """
    Calculate the multiclass AUC score using one-vs-all strategy
    :param y_true: true labels (one-hot-encoded)
    :param y_pred: predicted probabilities (one-hot-encoded)
    :param average: averaging strategy (default is 'macro')
    :return: multiclass AUC score
    """
    # calculate the AUC for each class
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc_scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    print(auc_scores)
    
    # calculate the average AUC across all classes
    if average == "macro":
        return np.mean(auc_scores)
    elif average == "micro":
        return roc_auc_score(y_true.ravel(), y_pred.ravel())
    else:
        return roc_auc_score(y_true, y_pred, average=average)



multiclass_roc_auc_score(truelabels, predictions, average="macro")
print('the overall area under curve',multiclass_roc_auc_score(truelabels, predictions, average="macro"))
