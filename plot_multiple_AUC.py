
import torch
import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"





class convNext(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        self.calssifier =nn.Sequential(nn.Flatten(1, -1),
                                       nn.Dropout(p=0.1),
                                       #nn.Linear(in_features=262144, out_features=2)) # 1024*7*7 = 50176
                                       nn.Linear(in_features=1024, out_features=2))

    def forward(self, x):
        feature = self.feature(x) # this feature we can use when doing stnad.Att
        
        print(feature.shape)
        flatten_featur = feature.reshape(feature.size(0), -1) #this we need to plot tsne
        x = self.calssifier(feature)
        return flatten_featur
        #return #x

    
model =convNext().to(device=device,dtype=torch.float32)

img = torch.rand(1,3,224,224)
out = model(img)
print(out.shape)



# Define your Transformer model
class TransformerModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(TransformerModel, self).__init__()
        
        # Load a pre-trained CNN for image feature extraction
        convNext = models.convnext_base(pretrained=True)
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        
        # Define tabular feature processing
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model=1024+64, nhead=8, num_encoder_layers=2)
        
        # Output layer
        self.output_layer = nn.Linear(192, num_classes)
        
    def forward(self, tabular_data, image_data):
        
        feature = self.feature(image_data) # this feature we can use when doing stnad.Att
        
        print(feature.shape)
        flatten_featur = feature.reshape(feature.size(0), -1)
        # Extract image features
        #image_features = self.image_encoder(image_data)
        
        # Process tabular data
        tabular_features = self.tabular_encoder(tabular_data)
        
        # Concatenate or stack image and tabular features
        combined_features = torch.cat((tabular_features, flatten_featur), dim=1)
        
        # Add positional encodings if needed
        
        # Pass through the Transformer layers
        transformer_output = self.transformer(combined_features)
        
        # Final output layer
        output = self.output_layer(transformer_output)
        
        return output

# Instantiate the model
model = TransformerModel(num_tabular_features=3, num_classes=2)

# Assuming you have your tabular and image data as tensors
tabular_data = torch.randn(2, 3)
image_data = torch.randn(2, 3, 224, 224)  # Assuming RGB images of size 224x224

# Forward pass
output = model(tabular_data, image_data)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop and data loading will depend on your specific dataset.







# First, save the ground truth and the prediction when testing the model to a dataframe 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
path = r"E:\BMVC_rebuttel\TruePred_data_CNN.csv"
data = pd.read_csv(path)
path2 = r"E:\BMVC_rebuttel\TruePred_cross_OA.csv"
data2 = pd.read_csv(path2)

path3 =  r"E:\BMVC_rebuttel\TruePred_stanA_mlp.csv"
data3 = pd.read_csv(path3)

path4 =  r"E:\BMVC_rebuttel\TruePred_cross_od.csv"
data4 = pd.read_csv(path4)

path5 =  r"E:\BMVC_rebuttel\TruePred_FOAA.csv"
data5 = pd.read_csv(path5)


fpr1, tpr1, thresholds1 = roc_curve(data['true'], data['pred'])
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, thresholds2 = roc_curve(data2['true'], data2['pred'])
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, thresholds3 = roc_curve(data3['true'], data3['pred'])
roc_auc3 = auc(fpr3, tpr3)


fpr4, tpr4, thresholds4 = roc_curve(data4['true'], data4['pred'])
roc_auc4 = auc(fpr4, tpr4)


fpr5, tpr5, thresholds5 = roc_curve(data5['true'], data5['pred'])
roc_auc5 = auc(fpr5, tpr5)


fig = plt.figure(figsize=(8,8))

plt.plot(fpr1, tpr1, color='green', lw=2, label=f'CNN-Only (AUC = {roc_auc1:.2f})')

# Plot the second AUC curve
plt.plot(fpr2, tpr2, color='navy', lw=2, label=f'CNN + MLP standardA (AUC = {roc_auc2:.2f})')

#plt.plot(fpr3, tpr3, color='red', lw=2, label=f'CNN branch(AUC = {roc_auc3:.2f})')

#plt.plot(fpr4, tpr4, color='green', lw=2, label=f'CNN branch(AUC = {roc_auc4:.2f})')

plt.plot(fpr5, tpr5, color='red', lw=3, label=f'FOAA (AUC = {roc_auc5:.2f})')
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
