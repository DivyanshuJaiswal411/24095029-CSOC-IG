import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc

class Hospital_Data(Dataset):
    def __init__(self,x_data,y_data):
        self.x=torch.tensor(x_data,dtype=torch.float32)
        self.y=torch.tensor(y_data,dtype=torch.float32)
        self.n_sample=x_data.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_sample

#defining parameters:-

input_size=7
hidden_size=15
learning_rate=0.0001
n_epoch=500


#Neural Network Code

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNetwork, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.linear1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        out=self.linear1(x)
        out=self.relu(out)
        out=self.linear2(out)
        out=self.sigmoid(out)
        return out

model=NeuralNetwork(input_size,hidden_size)

#data loading and processing 

df=pd.read_csv(r"C:\Users\Divyanshu\Downloads\KaggleV2-May-2016.csv\KaggleV2-May-2016.csv")

X=df.iloc[:,[5,7,8,9,10,11,12]].values.astype(np.float32)
y=(df.iloc[:,13].values=="Yes").astype(np.float32).reshape(-1,1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Dataset and DataLoader

train_dataset=Hospital_Data(X_train,y_train)
test_dataset=Hospital_Data(X_test,y_test)

train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=1000)
test_loader=DataLoader(dataset=test_dataset,shuffle=True,batch_size=1000)

# Compute positive weight
pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

# Custom Weighted BCELoss
def weighted_bce_loss(output, target):
    weights = torch.ones_like(target)
    weights[target == 1] = pos_weight
    return nn.BCELoss(weight=weights)(output, target)

#Loss and Optimisation 

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#training the model

length=len(train_dataset)
n_iters=math.ceil(length/1000)

for epoch in range(n_epoch):
    for i,(features,labels) in enumerate(train_loader):

        #forward pass

        output=model(features)
        loss = weighted_bce_loss(output, labels)

        #backward pass and optimization

        optimizer.zero_grad()               #empty the optimisation as it becomes stacked after every iteration
        loss.backward()
        optimizer.step()

        if (i+1)%20==0:
            print(f'Epoch:[{epoch+1}/{n_epoch}] , step:[{i+1}/{n_iters}], Loss:{loss.item():.4f}')

#testing the model

with torch.no_grad():
    y_pred_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    y_pred = (y_pred_probs >= 0.47).astype(int)
    y_true = y_test.astype(int)

accuracy = (y_pred == y_true).mean()
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

# Classification Report
print(classification_report(y_true, y_pred, target_names=["No", "Yes"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()    

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
pr_auc = auc(recall, precision)
print(f"PR-AUC (PyTorch): {pr_auc:.4f}")

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (PyTorch)')
plt.legend()
plt.grid(True)
plt.savefig("pr_curve_pytorch.png")  # Save to include in LaTeX
plt.show()