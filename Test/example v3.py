
"""
Created on Wed Oct 16 09:11:22 2024

@author: feiz
"""

# %% imports
import torch
import torch.nn as nn
import inspect

# Source TransformerJM code
import sys
sys.path.append("C:/research/TJM/TransformerJM/")
from Models.Transformer.TransformerJM import Transformer
from Models.Transformer.functions import (get_tensors, get_mask, init_weights, get_std_opt)
from Models.Transformer.loss import (long_loss, surv_loss)
# from Models.metrics import (AUC, Brier, MSE)
from Simulation.data_simulation_base import simulate_JM_base

print(inspect.getsource(surv_loss))

# %% Other Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% Global options
n_sim = 1
I = 1000
obstime = [0,1,2,3,4,5,6,7,8,9,10]
landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]
scenario = "none" # ["none", "interaction", "nonph"]

# %% Simulated data
data_all = simulate_JM_base(I=I, obstime=obstime, opt=scenario, seed=n_sim)
data = data_all[data_all.obstime <= data_all.time]
data.iloc[:, :6].head(40)

data['time'] += 0.1 # survival time greater than last obstime, model can still run

data['time'] -= 0.2 # survival time greater than last obstime, model can still run


# %% Using data simulated from R
I = 5000
data = data_full = pd.read_csv("Test/r_data.csv")
data.iloc[:, :6].head(20)
data.shape
# data['time'] += 0.5

# %% remove last obs time
# Find the index of the row with maximum obstime for each id
max_idx = data.groupby('id')['obstime'].idxmax()
# Drop those rows from the DataFrame
data_without_last = data.drop(max_idx)
print(data_without_last.shape)
data = data_without_last
data['time'] = data['ctstime']
data = data[data['time'] >= 0.5]
# data['time'] = data.groupby('id')['obstime'].transform('max')
data.iloc[:, :8].head(20)


# %% split train/test
I = data["id"].nunique()
print("Number of unique ids:", I)
random_id = range(I) #np.random.permutation(range(I))
train_id = random_id[0:int(0.7*I)]
test_id = random_id[int(0.7*I):I]

train_data = data[data["id"].isin(train_id)]
test_data = data[data["id"].isin(test_id)]

# %% Scale data using Min-Max Scaler
# minmax_scaler = MinMaxScaler(feature_range=(-1,1))
# train_data.loc[:,["Y1","Y2","Y3"]] = minmax_scaler.fit_transform(train_data.loc[:,["Y1","Y2","Y3"]])
# test_data.loc[:,["Y1","Y2","Y3"]] = minmax_scaler.transform(test_data.loc[:,["Y1","Y2","Y3"]])

# %% Train model
torch.manual_seed(0)

model = Transformer(d_long=1, d_base=1, d_model=32, nhead=4,
                    num_decoder_layers=7)
model.apply(init_weights)
model = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_std_opt(optimizer, d_model=32, warmup_steps=200, factor=0.2)


n_epoch = 5
batch_size = 32


loss_values_train = []
loss_values_val = []

# %% run epochs
for epoch in range(n_epoch):
    model.train()
    running_loss_train = 0
    # Shuffle training ids
    negative_loss_found = False  # flag to break out of outer loop if needed
    train_id = np.random.permutation(train_id)
    for batch in range(0, len(train_id), batch_size):
        optimizer.zero_grad()

        indices = train_id[batch:batch+batch_size]
        batch_data = train_data[train_data["id"].isin(indices)]

        batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(batch_data.copy(),
                                                                                     long=["Y"],base=["X1"], 
                                                                                     obstime="obstime")
        batch_long_inp = batch_long[:,:-1,:]
        batch_long_out = batch_long[:,1:,:]
        batch_base = batch_base[:,:-1,:]
        batch_mask_inp = get_mask(batch_mask[:,:-1])
        batch_mask_out = batch_mask[:,1:].unsqueeze(2)

        # print(obs_time[:,:-1].shape)
        yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp,
                                     obs_time[:,:-1], obs_time[:,1:])
        loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
        loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
        loss = loss1 + loss2

        # Check if loss2 is negative
        if loss2.item() < 0:
            print("Negative surv_loss encountered:", loss2.item())
            negative_loss_found = True
            break  # exit the batch loop

        loss = loss1 + loss2
        loss.backward()
        scheduler.step()
        running_loss_train += loss.item()  # use .item() to get a scalar value

    # Optionally, break out of the epoch loop if negative loss was found
    if negative_loss_found:
        print("Stopping training due to negative surv_loss.")
        break

    running_loss_train = running_loss_train/ len(train_id)    
    loss_values_train.append(running_loss_train)

    # Validation loop
    model.eval()
    running_loss_val = 0
    with torch.no_grad():
        for batch in range(0, len(test_id), batch_size):
            indices = test_id[batch:batch+batch_size]
            batch_data = test_data[test_data["id"].isin(indices)]

            batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(batch_data.copy(),
                                                                                         long=["Y"],base=["X1"], 
                                                                                         obstime="obstime")
            batch_long_inp = batch_long[:,:-1,:]
            batch_long_out = batch_long[:,1:,:]
            batch_base = batch_base[:,:-1,:]
            batch_mask_inp = get_mask(batch_mask[:,:-1])
            batch_mask_out = batch_mask[:,1:].unsqueeze(2)

            yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp,
                                         obs_time[:,:-1], obs_time[:,1:])
            loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
            loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
            loss_val = loss1 + loss2

            running_loss_val += loss_val.item()
            
            
    running_loss_val = running_loss_val/ len(test_id)
    loss_values_val.append( running_loss_val)

    print(f"Epoch {epoch+1}/{n_epoch} - Train Loss: {running_loss_train:.4f} - Val Loss: {running_loss_val:.4f}")



# %% Plot both training and validation losses without normalization
plt.figure(figsize=(10, 6))
plt.plot(loss_values_train, 'b-', label='Training Loss')
plt.plot(loss_values_val, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss_plot.png")  # Save as PNG file
plt.show()
# Save the plot to a file
# plt.savefig("plot.png")  # Save as PNG file


# %% Prediction
landmark_times
LT = 1
pred_times = [x+LT for x in pred_windows]

# Only keep subjects with survival time > landmark time
tmp_data = test_data.loc[test_data["time"]>LT,:]

# Only keep longitudinal observations <= landmark time
tmp_data = tmp_data.loc[tmp_data["obstime"]<=LT,:]


tmp_long, tmp_base, tmp_mask, e_tmp, t_tmp, obs_time_1 = get_tensors(tmp_data.copy(),long=["Y"],base=["X1"], 
                                                                                         obstime="obstime")
train_long, train_base, train_mask, e_train, t_train, obs_time = get_tensors(train_data.copy(),long=["Y"],base=["X1"], 
                                                                                         obstime="obstime")


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
    if obs_time_1.shape[1] != dec_long.shape[1]:
        obs_time_1 = torch.cat([obs_time_1, torch.zeros(obs_time_1.shape[0], dec_long.shape[1] - obs_time_1.shape[1])], dim=1)

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

# auc, iauc = AUC(surv_pred, e_tmp.numpy(), t_tmp.numpy(), np.array(pred_times))
# auc
# iauc

# bs, ibs = Brier(surv_pred, e_tmp.numpy(), t_tmp.numpy(),
#                   e_train.numpy(), t_train.numpy(), LT, np.array(pred_windows))

# bs
















# %%
