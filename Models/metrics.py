import torch
import numpy as np
import warnings
import os

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r.source("C:/research/TJM/TransformerJM/Models/AUC_BS.r")
AUC_R = ro.globalenv['AUC']
Brier_R = ro.globalenv['Brier']


def get_integrated(x, times):
    return np.trapz(x,times) / (max(times)-min(times))

def AUC(x, event, time, pred_times):
    auc = AUC_R(surv=x, event=event, time=time, predtimes=pred_times)[0]
    iauc = get_integrated(auc, pred_times)
    return auc, iauc

def Brier(x, event, time, event_train, time_train, LT, pred_windows):
    bs = Brier_R(surv=x, event=event, time=time,
                 event_train=event_train, time_train=time_train,
                 LT = LT, DeltaT=pred_windows)
    ibs = get_integrated(bs, pred_windows)
    return bs, ibs


def MSE(y, yhat):
    mse = np.square(y-yhat)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        mse = np.nanmean(mse, axis=1) # average over time
    mse = np.nanmean(mse, axis=0) # average over subj
    return mse


import numpy as np

# Generate random survival probabilities
surv_test = np.random.rand(282, 3)
surv_test_r = ro.r.matrix(ro.FloatVector(surv_test.flatten()), nrow=282, ncol=3)

# Create event indicators (1s)
e_tmp_test = np.ones(282, dtype=int)

# Create time to event or censoring times
t_tmp_test = np.array([7, 7, 4, 6, 7, 6, 4, 10, 10, 4, 6, 6, 7, 9, 2, 10, 5, 6, 6, 10, 10, 6, 10, 7,
                       10, 8, 4, 5, 10, 4, 10, 9, 10, 10, 10, 7, 7, 9, 4, 10, 9, 4, 9, 6, 10, 8, 3, 10,
                       10, 3, 7, 10, 10, 9, 9, 8, 2, 6, 6, 10, 10, 3, 4, 10, 8, 7, 10, 5, 5, 10, 10, 4,
                       8, 10, 6, 10, 10, 6, 7, 9, 9, 5, 10, 10, 7, 7, 9, 5, 10, 2, 7, 3, 6, 4, 6, 10,
                       7, 10, 6, 8, 7, 10, 2, 8, 10, 10, 7, 10, 8, 6, 10, 3, 2, 4, 2, 4, 3, 7, 6, 5,
                       9, 10, 3, 2, 5, 9, 10, 9, 6, 10, 8, 2, 10, 5, 10, 7, 7, 6, 5, 9, 3, 7, 10, 2,
                       5, 5, 4, 7, 4, 8, 10, 6, 6, 10, 7, 7, 9, 10, 8, 4, 10, 5, 5, 10, 7, 7, 10, 8,
                       5, 9, 10, 6, 7, 7, 10, 10, 3, 9, 10, 6, 10, 4, 2, 8, 10, 8, 8, 10, 3, 4, 9, 10,
                       2, 6, 10, 9, 10, 3, 6, 8, 5, 2, 9, 3, 3, 7, 10, 9, 10, 7, 4, 10, 5, 10, 4, 3,
                       7, 8, 5, 10, 10, 10, 6, 8, 3, 10, 4, 10, 10, 2, 10, 4, 10, 10, 5, 10, 4, 10, 10, 6,
                       9, 9, 5, 7, 2, 10, 10, 7, 2, 8, 5, 4, 8, 4, 5, 8, 7, 10, 7, 6, 5, 5, 2, 8,
                       9, 2, 10, 2, 10, 6, 10, 10, 4, 8, 10, 10, 8, 3, 8, 10, 8, 10])

# Define prediction times
predtimes = np.array([2, 3, 4])

AUC_R(surv=surv_test, event=e_tmp_test, time=t_tmp_test, predtimes=predtimes)