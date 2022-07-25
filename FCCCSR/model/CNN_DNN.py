import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DataProcessTest import getTest
from model.DataProcessTrain import getTrain
import torch.utils.data as Data
from transformers import WarmupLinearSchedule
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
torch.manual_seed(23)
np.random.seed(23)
torch.manual_seed(1)
#Define the cnn_dnn model
def cnn_dnn():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    np.random.seed(23)
    from sklearn.preprocessing import MinMaxScaler
    np.random.seed(23)
    from sklearn.preprocessing import MinMaxScaler
    torch.manual_seed(1)
    #loading training set
    X_train, y_train = getTrain()
    #loading testing set
    X_test, y_test = getTest()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #numpy->tensor
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    #define  MyDeep() 
    class MyDeep(nn.Module):
        def __init__(self, conv1_size, maxp1_size, conv2_size):
            super(MyDeep, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=27, kernel_size=conv1_size)
            self.maxpool1 = nn.MaxPool1d(kernel_size=maxp1_size)
            self.conv2 = nn.Conv1d(27, 15, kernel_size=conv2_size)
            self.fl = torch.nn.Flatten()
            self.dropout = nn.Dropout(0.5)
            self.liner1 = nn.Linear(15 * int((((76 - conv1_size + 1) / (maxp1_size)) - conv2_size + 1)), 260)
            self.liner2 = nn.Linear(260, 70)
            self.liner3 = nn.Linear(70, 1)
        def forward(self, x):
            length = len(x)
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = F.relu(self.conv2(x))
            x = self.fl(x)
            x = F.relu(self.liner1(x))
            x = F.relu(self.liner2(x))
            x = self.dropout(x)
            x = self.liner3(x)
            x = torch.sigmoid(x)
            return x
    torch.manual_seed(1)
    #BATCH_SIZE
    BATCH_SIZE =108
    #encapsulated dataset
    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    # ---------------------------training model------------------------------------
    import torch.optim as optim
    #number of iterations
    N_EPOCHS = 100
    #convolution kernel size
    Conv1_size = 5
    Maxp1_size = 4
    Conv2_size = 3
    model = MyDeep(Conv1_size, Maxp1_size, Conv2_size).cuda()
    #optimizer
    optimizer = optim.Adam(model.parameters(), eps=5e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=6e-6, t_total=len(train_loader) * N_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    #count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    #ACC
    def binary_accuracy(preds, y):
        rounded_preds = torch.round(preds)
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc
    #metric
    def metric(preds, y):
        a = preds.cpu().numpy()
        rounded_preds = torch.round(preds)
        TN, FP, FN, TP = confusion_matrix(y.cpu().numpy(), rounded_preds.cpu().numpy()).ravel()
        SN = recall_score(y.cpu().numpy(), rounded_preds.cpu().numpy())
        SP = TN / (TN + FP)
        MCC = matthews_corrcoef(y.cpu().numpy(), rounded_preds.cpu().numpy())
        ACC = (TP + TN) / (TP + TN + FN + FP)
        return SN, SP, ACC, MCC,a
    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for i, batch in enumerate(iterator, 0):
            x_data, x_label = batch
            optimizer.zero_grad()
            x_data = x_data.unsqueeze(1).float()
            predictions = model(x_data.cuda()).squeeze(1)
            loss = criterion(predictions, x_label.cuda().float())
            acc = binary_accuracy(predictions, x_label.cuda().float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    #test function
    def Test(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        epoch_SN = 0
        epoch_SP = 0
        epoch_ACC = 0
        epoch_MCC = 0
        model.eval()
        with torch.no_grad():
            cnn_pro=[]
            cnn_y=[]
            for i, batch in enumerate(iterator, 0):
                test, t_lable = batch
                cnn_y=cnn_y+t_lable.tolist()
                test = test.unsqueeze(1).float()
                predictions = model(test.cuda()).squeeze(1)
                loss = criterion(predictions, t_lable.cuda().float())
                acc = binary_accuracy(predictions, t_lable.cuda())
                SN, SP, ACC, MCC,CNN_DNN_Pro = metric(predictions, t_lable.cuda())
                CNN_DNN_Pro=CNN_DNN_Pro.tolist()
                cnn_pro=cnn_pro+CNN_DNN_Pro
                epoch_SN += SN
                epoch_SP += SP
                epoch_ACC += ACC
                epoch_MCC += MCC
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_SN / len(iterator), epoch_SP / len(
            iterator), epoch_ACC / len(iterator), epoch_MCC / len(iterator),cnn_pro,cnn_y
    import time
    #calculating time
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    print('---------DNN_CNN_Training-------')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
     #Loading model parameters
    model.load_state_dict(torch.load('./parameter/tut1-model.pt'))
    print('---------DNN_CNN_Testing-------')
    test_loss, test_acc, test_sn, test_sp, test_ACC, test_mcc,CNN_DNN_Pro,cnn_y= Test(model, test_loader, criterion)
    print(f'test_sn: {test_sn*100:.2f}% | test_sp: {test_sp*100:.2f}%')
    print(f'test_mcc: {test_mcc:.4f} | test_ACC: {test_ACC * 100:.2f}%')
    return CNN_DNN_Pro,cnn_y
