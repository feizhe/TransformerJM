#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:11:22 2024

@author: feiz
"""

# Pytorch
import torch
import torch.nn as nn

# Source TransformerJM code
import sys
sys.path.append("C:/Users/jgmea/research/transf/TransformerJM")


from Models.Transformer.TransformerJM import Transformer
from Models.Transformer.functions import (get_tensors, get_mask, init_weights, get_std_opt)
from Models.Transformer.loss import (long_loss, surv_loss)
#from Models.metrics import (AUC, Brier, MSE)
from Simulation.datsim import simulate_JM_base2

# Other Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
from pathlib import Path
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global options
n_sim = 1
I = 5000
obstime = np.linspace(0,10,num=21)#[0,1,2,3,4,5,6,7,8,9,10]
landmark_times = np.linspace(1,4.5,num=8)
pred_windows = np.linspace(0.5,4,num=8)
scenario = "none" # ["none", "interaction", "nonph"]

# data from the simulate_JM_base2 function in datsim.py
#data_all = simulate_JM_base2(I=I, obstime=obstime, opt=scenario, seed=n_sim)
#data = data_all[data_all.obstime <= data_all.time]
# print(data.shape)
#(7647, 15)

# data from the r_data.csv generated by jmbayes_datasimulator.R
data_r = pd.read_csv("r_data.csv")

data_r.head()
data_r.shape

## split train/test
random_id = range(I) #np.random.permutation(range(I))
train_id = random_id[0:int(0.7*I)]
test_id = random_id[int(0.7*I):I]

#training data from TransformerJM datsim.py
#train_data = data[data["id"].isin(train_id)]
# training data from the r_data.csv generated by jmbayes_datasimulator.R
train_data = data_r[data_r["id"].isin(train_id)]



# test data from TransformerJM datsim.py
#test_data = data[data["id"].isin(test_id)]
# training data from the r_data.csv generated by jmbayes_datasimulator.R
test_data = data_r[data_r["id"].isin(test_id)]


## Scale data using Min-Max Scaler
#minmax_scaler = MinMaxScaler(feature_range=(-1,1))
#train_data.loc[:,["Y"]] = minmax_scaler.fit_transform(train_data.loc[:,["Y"]])
#test_data.loc[:,["Y"]] = minmax_scaler.transform(test_data.loc[:,["Y"]])

## Train model
torch.manual_seed(0)

model = Transformer(d_long=1, d_base=1, d_model=32, nhead=4,
                    num_decoder_layers=7)
model.apply(init_weights)
model = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_std_opt(optimizer, d_model=32, warmup_steps=200, factor=0.2)


n_epoch = 25
batch_size = 32   #####
start = time.time()
loss_values = []
val_loss_values = []
for epoch in range(n_epoch):
    running_loss = 0
    train_id = np.random.permutation(train_id)
    val_id = np.random.permutation(test_id)
    print(epoch)
    for batch in range(0, len(train_id), batch_size):
        optimizer.zero_grad()

        indices = train_id[batch:batch+batch_size]
        batch_data = train_data[train_data["id"].isin(indices)]
        #print(batch_data.shape)
        batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(batch_data.copy(),long=["Y"],base=["X1"])
        batch_long_inp = batch_long[:,:-1,:]
        batch_long_out = batch_long[:,1:,:]
        batch_base = batch_base[:,:-1,:]
        batch_mask_inp = get_mask(batch_mask[:,:-1])
        batch_mask_out = batch_mask[:,1:].unsqueeze(2)


        # at this point, some batches return missing values for predicted yhat_long
        yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp,
                     obs_time[:,:-1], obs_time[:,1:])
        loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
        #print(loss1)
        loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
        loss = loss1 + loss2
        loss.backward()
        scheduler.step()
        running_loss += loss
    loss_values.append(running_loss.tolist())
    
    test_id = np.random.permutation(test_id)
    running_val_loss = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch in range(0, len(test_id), batch_size):
            # Prepare validation batch data
            indices = test_id[batch:batch + batch_size]
            batch_data = test_data[test_data["id"].isin(indices)]
            batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(
                batch_data.copy(), long=["Y"], base=["X1"])
            batch_long_inp = batch_long[:, :-1, :]
            batch_long_out = batch_long[:, 1:, :]
            batch_base = batch_base[:, :-1, :]
            batch_mask_inp = get_mask(batch_mask[:, :-1])
            batch_mask_out = batch_mask[:, 1:].unsqueeze(2)

            # Forward pass
            yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp, obs_time[:, :-1], obs_time[:, 1:])

            # Compute validation loss
            val_loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
            val_loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
            val_loss = val_loss1 + val_loss2
           # val_loss.backward()
           # scheduler.step()

            # Accumulate validation loss
            running_val_loss += val_loss
    val_loss_values.append(running_val_loss.tolist())




print(time.time() - start)
plt.plot((loss_values-np.min(loss_values))/(np.max(loss_values)-np.min(loss_values)), 'b-') 
plt.xlabel('Iterations')
plt.ylabel('Normalized Loss')
plt.title('Training Loss')
print(loss_values)
# Save the plot to a file
plt.savefig("train_loss.png")  # Save as PNG file




landmark_times
LT = 4.5
pred_times = [x+LT for x in pred_windows]
pred_times
# Only keep subjects with survival time > landmark time
tmp_data = test_data.loc[test_data["time"]>LT,:]

# Only keep longitudinal observations <= landmark time
tmp_data = tmp_data.loc[tmp_data["obstime"]<=LT,:]


tmp_long, tmp_base, tmp_mask, e_tmp, t_tmp, obs_time_1 = get_tensors(tmp_data.copy(),long=["Y"],base=["X1"])
train_long, train_base, train_mask, e_train, t_train, obs_time = get_tensors(train_data.copy(),long=["Y"],base=["X1"])
# print(obs_time_1.shape)

base_0 = tmp_base[:,0,:].unsqueeze(1)       
long_0 = tmp_long
mask_T = torch.ones((long_0.shape[0],1), dtype=torch.bool)

dec_long = long_0
dec_base = base_0

long_pred = torch.zeros(long_0.shape[0],0,long_0.shape[2])
surv_pred = torch.zeros(long_0.shape[0],0,1)

model = model.eval()





for pt in pred_times:
    dec_base = base_0.expand([-1,dec_long.shape[1],-1])
    # Ensure obs_time_1 has the correct dimensions
   
    # Debugging prints to check dimensions
    if obs_time_1.shape[1] != dec_long.shape[1]:
        obs_time_1 = torch.cat([obs_time_1, torch.zeros(obs_time_1.shape[0], dec_long.shape[1] - obs_time_1.shape[1])], dim=1)

    # Call the model's decoder


    print(dec_base.shape)
    print(dec_long.shape)
    out = model.decoder(dec_long, dec_base, get_mask(tmp_mask), obs_time_1)
    out = model.decoder_pred(out[:,-1,:].unsqueeze(1), out,
          tmp_mask.unsqueeze(1), torch.tensor(pt))
    long_out = model.long(out)
    surv_out = torch.sigmoid(model.surv(out))

    long_pred = torch.cat((long_pred, long_out), dim=1)
    surv_pred = torch.cat((surv_pred, surv_out), dim=1)

    dec_long = torch.cat((dec_long, long_out), dim=1)
    tmp_mask = torch.cat((tmp_mask, mask_T), dim=1)
    obs_time = torch.cat((obs_time, torch.tensor(pt).expand([obs_time.shape[0],1])),dim=1)

long_pred = long_pred.detach().numpy()
surv_pred = surv_pred.squeeze().detach().numpy()
surv_pred = surv_pred.cumprod(axis=1)


print(surv_pred.shape)

print("surv_pred shape:", surv_pred.shape)
print("e_tmp:", e_tmp.shape)
#print(e_tmp.numpy().astype(int))
print("t_tmp:", t_tmp.shape)
#print(t_tmp.numpy())

# print(surv_pred)

#print("pred_times:", pred_times)  (2,3,4) interation 1 pred_times = 2
# auc, iauc = AUC(surv_pred, e_tmp.numpy().astype(int), t_tmp.numpy(), np.array(pred_times))
# auc
# iauc

# bs, ibs = Brier(surv_pred, e_tmp.numpy().astype(int), t_tmp.numpy(),
#                   e_train.numpy(), t_train.numpy(), LT, np.array(pred_windows))

# bs
#print(long_pred)
print("surv_pred type:", type(surv_pred))
print("long_pred type:", type(long_pred), long_pred.shape)
print(type(e_tmp))
print(type(t_tmp))

tmp_data.to_csv("tmp_data1.csv", index=False)
train_data.to_csv("train_data1.csv", index=False)
test_data.to_csv("test_data1.csv", index=False)
np.savetxt("surv_pred_1.csv", surv_pred, delimiter = ",")
#surv_pred.to_csv(, index=False)
#long_pred.to_csv("long_pred.csv", index=False)
#np.savetxt("long_pred.csv", long_pred, delimiter = ",")
event_tmp = pd.DataFrame(e_tmp.numpy().astype(int))
event_tmp.to_csv("event_tmp1.csv", index=False)
time_tmp = pd.DataFrame(t_tmp.numpy())
time_tmp.to_csv("time_tmp1.csv", index=False)

event_train = pd.DataFrame(e_train.numpy().astype(int))
event_train.to_csv("event_train1.csv", index=False)
time_train = pd.DataFrame(t_train.numpy())
time_train.to_csv("time_train1.csv", index=False)





# val_loss_values = []
# for epoch in range(n_epoch):
#     test_id = np.random.permutation(test_id)
#     running_val_loss = 0

#     with torch.no_grad():  # Disable gradient calculation
#         for batch in range(0, len(test_id), batch_size):
#             # Prepare validation batch data
#             indices = test_id[batch:batch + batch_size]
#             batch_data = test_data[test_data["id"].isin(indices)]
#             batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(
#                 batch_data.copy(), long=["Y"], base=["X1"])
#             batch_long_inp = batch_long[:, :-1, :]
#             batch_long_out = batch_long[:, 1:, :]
#             batch_base = batch_base[:, :-1, :]
#             batch_mask_inp = get_mask(batch_mask[:, :-1])
#             batch_mask_out = batch_mask[:, 1:].unsqueeze(2)

#             # Forward pass
#             yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp, obs_time[:, :-1], obs_time[:, 1:])

#             # Compute validation loss
#             val_loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
#             val_loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
#             val_loss = val_loss1 + val_loss2
#            # val_loss.backward()
#            # scheduler.step()

#             # Accumulate validation loss
#             running_val_loss += val_loss
#     val_loss_values.append(running_val_loss.tolist())

# Plot validation loss
plt.plot((val_loss_values-np.min(val_loss_values))/(np.max(val_loss_values)-np.min(val_loss_values)), 'r-') 
plt.xlabel('Iterations')
plt.ylabel('Normalized  Loss')
plt.title('Training and Validation Loss')
print(val_loss_values)
plt.savefig("val_loss.png")  # Save as PNG file

