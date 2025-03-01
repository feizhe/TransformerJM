
# def surv_loss(surv_pred, mask, event):

# %%
surv_pred = yhat_surv
mask = batch_mask
event = batch_e

mask = mask.numpy()
event = event.numpy()
mask_out = mask[:,1:]
mask_rev = mask_out[:,::-1]
event_time_index = mask_out.shape[1] - np.argmax(mask_rev, axis=1) - 1

e_filter = np.zeros([mask_out.shape[0],mask_out.shape[1]])
for row_index, row in enumerate(e_filter):
    if event[row_index]:
        row[event_time_index[row_index]] = 1
s_filter = mask_out - e_filter

# Debug: Output row sums of e_filter
print("Row sums of e_filter:", np.sum(e_filter, axis=1), flush=True)
# e_filter_row_sums = np.atleast_1d(np.sum(e_filter, axis=1))

s_filter = torch.tensor(s_filter)
e_filter = torch.tensor(e_filter)

surv_pred = surv_pred.squeeze()
nll_loss = torch.log(surv_pred)*s_filter + torch.log(1-surv_pred)*e_filter
nll_loss = nll_loss.sum() / mask_out.sum()

#nll_loss = nll_loss.sum(dim=1) / torch.tensor(mask_out.sum(axis=1))
#nll_loss = nll_loss.mean()

print(-nll_loss)

# %%
