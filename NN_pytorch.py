import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

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
hidden_size=7
learning_rate=0.00001
n_epoch=100


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

#split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Dataset and DataLoader

train_dataset=Hospital_Data(X_train,y_train)
test_dataset=Hospital_Data(X_test,y_test)

train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=1000)
test_loader=DataLoader(dataset=test_dataset,shuffle=True,batch_size=1000)

#Loss and Optimisation 

l=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#training the model

length=len(train_dataset)
n_iters=math.ceil(length/1000)

for epoch in range(n_epoch):
    for i,(features,labels) in enumerate(train_loader):

        #forward pass

        output=model(features)
        loss=l(output,labels)

        #backward pass and optimization

        optimizer.zero_grad()               #empty the optimisation as it becomes stacked after every iteration
        loss.backward()
        optimizer.step()

        
        print(f'Epoch:[{epoch+1}/{n_epoch}] , step:[{i+1}/{n_iters}], Loss:{loss.item():.4f}')

#testing the model

with torch.no_grad():
    n_correct=0
    n_sample=0
    for test_features,y in test_loader:
        y_hat=model(test_features)

    _,predicted =torch.max(y_hat.data,1)
    n_sample+=y.size(0)
    n_correct+=(predicted==y).sum().item()

accuracy=(n_correct/n_sample)
print(f"Accuracy of the network by testing on 20% of the data:{accuracy}")
