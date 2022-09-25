#!/usr/bin/env python
# coding: utf-8

# In[1]:


import load_data
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from Classifier import LSTMClassifier
import pickle


# In[2]:


# Hyperparameters, feel free to tune

batch_size = 27
output_size = 9   # number of class
hidden_size = 50  # LSTM output size of each time step
input_size = 12
basic_epoch = 300
Adv_epoch = 100
Prox_epoch = 100
epsilon = 1e-3


# In[3]:


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


# In[4]:


# Training model
def train_model(model, train_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        input = batch[0]
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        r = 0
        optim.zero_grad()
        prediction = model(input, r, batch_size = input.size()[0], mode = 'plain')
        loss = loss_fn(prediction, target)
        if mode == 'AdvLSTM':


            ''' Add adversarial training term to loss'''
            r = compute_perturbation(loss, model)
            optim.zero_grad()
            prediction = model(input, r, batch_size = input.size()[0], mode = mode)
            target = target.repeat(2)
            loss = loss_fn(prediction, target)
            
#         elif mode == 'ProxLSTM':

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/(input.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


# In[5]:


# Test model
def eval_model(model, test_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    r = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            input = batch[0]
            target = batch[1]
            target = torch.autograd.Variable(target).long()
            prediction = model(input, r, batch_size=input.size()[0], mode = 'plain')
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects.double()/(input.size()[0])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)


# In[6]:


def compute_perturbation(loss, model):

    '''need to be implemented'''
    # Use autograd
    g = torch.autograd.grad(loss, model.lstm_input)[0]
    L2_g = F.normalize(g, dim=1)
    r = g / L2_g
    return r


# In[8]:


''' Training basic model '''

train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)


model = LSTMClassifier(batch_size, output_size, hidden_size, input_size, epsilon)
loss_fn = F.cross_entropy
acc_list = []

# for epoch in range(basic_epoch):
#         optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, 
#                 weight_decay=1e-3)
#         train_loss, train_acc = train_model(model, train_iter, mode = 'plain')
#         val_loss, val_acc = eval_model(model, test_iter, mode ='plain')
#         acc_list.append(val_acc)
#         print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
        
# with open("test_acc_basic_lstm.pkl", "wb") as f:
#     pickle.dump(acc_list, f)        
        
        
        


# In[10]:



''' Save and Load model'''

# 1. Save the trained model from the basic LSTM
# torch.save(model.state_dict(), "basic_lstm.pt")
# # 2. load the saved model to Prox_model, which is an instance of LSTMClassifier
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size, epsilon)    
Prox_model.load_state_dict(torch.load("basic_lstm.pt"))  

# # 3. load the saved model to Adv_model, which is an instance of LSTMClassifier
# Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size, epsilon)    
# Adv_model.load_state_dict(torch.load("basic_lstm.pt")) 


# In[ ]:


# ''' Training Prox_model'''
for epoch in range(Adv_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM')
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')


# In[12]:



# ''' Training Adv_model'''

# for epoch in range(Prox_epoch):
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
#     train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM')
#     val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='plain')
#     print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')


# In[ ]:





# In[ ]:




