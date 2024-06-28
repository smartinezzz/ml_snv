

import pandas as pd
import csv
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

samp_list=['SAMN13012145','SAMN13012146','SAMN13012147','SAMN13012148','SAMN13012149','SAMN13012150','SAMN12799275','SAMN12799274','SAMN12799273','SAMN12799270','SAMN12799269','SAMN12799266','SAMN12799264','SAMN12799263','SAMN15453063','SAMN15453064','SAMN12799259','SAMN12799258','SAMN12799257','SAMN15453062','SAMN16086829','SAMN16086830']

for i in range(len(samp_list)):
    this_samp = samp_list[i]
    scread_ref = pd.read_csv(this_samp+'/'+this_samp+'_scread_ref.csv')
    scread_var = pd.read_csv(this_samp+'/'+this_samp+'_scread_var.csv')
    targets = pd.read_csv(this_samp+'/'+this_samp+'_gsnreads.csv')
    targets.replace('germline', 1, inplace=True) 
    targets.replace('somatic', 2, inplace=True) 
    targets.replace('novel', 3, inplace=True)
    targets = targets[~mask]
    scread_ref = scread_ref[~mask]
    scread_var = scread_var[~mask]

    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    #X, y = make_classification(n_samples=77, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(scread_var.iloc[:,1:], targets['germsomnov'], stratify=targets['germsomnov'],
                                                        random_state=1)
    clf = MLPClassifier(hidden_layer_sizes=(20,),random_state=1, max_iter=300).fit(X_train, y_train)
    #clf.predict_proba(X_test[:1])
    print(clf.score(X_test, y_test))
    print(y_train.unique())



    nrow,ncol = X_train.shape

    class Model(nn.Module):
        # Input layer (# cells in the dataset) -->
        # Hidden Layer -->
        # output (2 classes)
    
        def __init__(self, in_features=ncol,h1=100,h2=20,h3=10,h4=10,h5=5,h6=5,h7=5,out_features=2):#,h2=100,h3=80,h4=80,h5=70,h6=70,h7=60,h8=60,h9=50,h10=50,h11=50,h12=30,h13=30,h14=20,h15=17,h16=15,h17=10,h18=10,h19=5,h20=5,out_features=2):
    #    def __init__(self, in_features=ncol,h1=100,h2=20,out_features=3):#,h3=80,h4=80,h5=70,h6=70,h7=60,h8=60,h9=50, h10=50,h11=50,h12=30,h13=30,h14=20,h15=17,h16=15,h17=10,h18=10,h19=5,h20=5,out_features=3):
            super().__init__() # instatiate our nn.Module
            #self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(in_features,h1)
            self.fc2 = nn.Linear(h1,h2)
            self.fc3 = nn.Linear(h2,h3)
            self.fc4 = nn.Linear(h3,h4)
            self.fc5 = nn.Linear(h4,h5)
            self.fc6 = nn.Linear(h5,h6)
            self.fc7 = nn.Linear(h6,h7)
            #self.fc8 = nn.Linear(h7,h8)
            #self.fc9 = nn.Linear(h8,h9)
            #self.fc10 = nn.Linear(h9,h10)
            ##self.fc11 = nn.Linear(h10,h11)
            #self.fc12 = nn.Linear(h11,h12)
            #self.fc13 = nn.Linear(h12,h13)
            ##self.fc14 = nn.Linear(h13,h14)
            #self.fc15 = nn.Linear(h14,h15)
            #self.fc16 = nn.Linear(h15,h16)
            #self.fc17 = nn.Linear(h16,h17)
            #self.fc18 = nn.Linear(h17,h18)
            #self.fc19 = nn.Linear(h18,h19)
            #self.fc20 = nn.Linear(h19,h20)
            self.out = nn.Linear(h7,out_features)
        
        def forward(self,x):
            #x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            #x = F.relu(self.fc8(x))
            #x = F.relu(self.fc9(x))
            #x = F.relu(self.fc10(x))
            #x = F.relu(self.fc11(x))
            #x = F.relu(self.fc12(x))
            #x = F.relu(self.fc13(x))
            #x = F.relu(self.fc14(x))
            #x = F.relu(self.fc15(x))
            #x = F.relu(self.fc16(x))
            ##x = F.relu(self.fc17(x))
            #x = F.relu(self.fc18(x))
            #x = F.relu(self.fc19(x))
            #x = F.relu(self.fc20(x))
            x = self.out(x)
            return x
        
    #torch.manual_seed(41)
    model=Model(ncol,2)

    #%% load data into model format
    X_train = torch.FloatTensor(X_train.to_numpy())
    X_test = torch.FloatTensor(X_test.to_numpy())

    y_train = torch.LongTensor(y_train.to_numpy())
    y_test = torch.LongTensor(y_test.to_numpy())


    #%% pytorch

    #Select model to measure the error
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    #Choose Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=0.01)

    epoch = 100
    losses = []
    for i in range(epoch):
        #forward
        y_pred = model.forward(X_train)
        loss = criterion(y_pred,y_train)
        losses.append(loss.detach().numpy())
        # print every 10th epoch
        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss}')
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    break
    # %%
    with torch.no_grad(): #turn off backprop
        y_eval=model.forward(X_test)
        loss=criterion(y_eval,y_test)
    print(loss)
    
    
    
    
    
    correct=0
    y_vals = list()
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val=model.forward(data)
            y_vals.append(y_val.argmax().item())
            print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')
        
            if y_val.argmax().item() == y_test[i]:
                correct +=1
    print(f'We got {correct} correct out of {i}')

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Compute the confusion matrix
    cm = confusion_matrix(y_test.numpy(), y_val)

    # Display using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
