import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random 
import time
image_dir="handsign"
letters = ["A","B","D","G","O","P","V","W","Y"]
s=time.time()
def dataloader(type_dir):
    data=[]
    path_=os.path.join(image_dir,type_dir)
    for letter in letters:
        path=os.path.join(path_,letter)
        label=letters.index(letter)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            resized_array=cv2.resize(img_array,(28,28))
            data.append([resized_array,label])
    random.seed(0)
    random.shuffle(data)
    X,Y=[],[]
    for x,y in data:
        X.append(x)
        Y.append(y)
    X=torch.Tensor(np.expand_dims(np.array(X),-1))
    Y=torch.LongTensor(np.array(Y))
    return X,Y
X_train,y_train=dataloader('train')
X_val,y_val=dataloader('test')
print("Data Loaded!")
print("time taken to load: "+str(time.time()-s))
class HandSignModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img=nn.Sequential(
            nn.Conv2d(1,64,3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(5*5*128,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,9),
            nn.BatchNorm1d(9),
        )
    def forward(self,X):
        return self.img(X.float())
torch.manual_seed(0)
model=HandSignModel()
cost_f=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)#,weight_decay=0.005)
decay=LambdaLR(optimizer, lr_lambda= lambda i: 0.95**i)
epoch=10
batch_size=1000
costs=[]
s=time.time()
for i in range(1,epoch+1):
    total_cost=0
    for j in range(9):
        X_mini=X_train[j*batch_size:(j+1)*batch_size]
        y_mini=y_train[j*batch_size:(j+1)*batch_size]
        y_pred = model(X_mini.permute(0,3,1,2))
        loss = cost_f(y_pred, y_mini)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_cost+=loss
    decay.step()
    costs.append(total_cost/batch_size)
    print("Cost after epoch "+str(i)+" = "+str((total_cost/batch_size).item()))
print("Training time taken: "+str(time.time()-s))
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.show()
def predict(X,y):
    correct=0
    model.eval()
    y_pred = model.forward(X.permute(0,3,1,2))
    pred=F.softmax(y_pred,dim=1)
    pred=torch.argmax(pred,axis=1)
    correct+=(y == pred).sum()
    return correct
torch.save(model.state_dict(),"trained.pth")
print("Accuracy on Train set is "+str(np.array(predict(X_train,y_train))*100/9000))
print("Accuracy on Test set is "+str(test:=np.array(predict(X_val,y_val))*100/1800))