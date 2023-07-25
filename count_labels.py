#To count how many of each class we have in our split 
y_train = train_dataset.dataset.LABEL[train_dataset.indices]
from collections import Counter
class_count= Counter(y_train) # y_true must be your labels
class_count
