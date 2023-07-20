import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, auc_curve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def eval_plt(test_loader,model,device):
    
    truelabels = []
    predictions = []
    proba = []
    pre = []
    
    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            data = data.to(device=device,dtype=torch.float)
            target = target.to(device=device,dtype=torch.float)
            truelabels.extend(target.cpu().numpy())
    
            output = model(data)
            probs = F.softmax(output, dim=1)[:, 1]# assuming logits has the shape [batch_size, nb_classes]
            #top_p, top_class = prob.topk(1, dim = 1)
            preds = torch.argmax(output, dim=1) # this to plot the confusion matrix
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            proba.extend(probs.cpu().numpy())
            pre.extend(preds.cpu().numpy())
            from sklearn.metrics import f1_score
        print('F1-score micro for MLP classifer:')
        print(f1_score(truelabels, predictions, average='micro'))
        print(classification_report(truelabels, predictions))
            
        cm = confusion_matrix(truelabels, pre)
        classes= ['GradeII', 'GradeIII', 'GradeIV']
        tick_marks = np.arange(len(classes))
        
        df_cm = pd.DataFrame(cm, index = classes, columns = classes)
        plt.figure(figsize = (7,7))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel("Predicted label", fontsize = 20)
        plt.ylabel("Ground Truth", fontsize = 20)
        plt.show()
