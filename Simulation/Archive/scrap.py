import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as optimize

I = 1000
obstime = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
opt="none"
seed = 1
#### Longitudinal submodel ####
beta0 = 1.5
beta1 = 2
betat = 1.5
b_var = 1
e_var = 1
rho = -0.2

# Variance-covariance matrix for random effects
#b_Sigma = np.diag(b_var)

# Covariate X1 (same for both submodels)
X1 = np.random.normal(3, 1, size=I)

# Random effects for longitudinal model
ranef = np.random.normal(0, b_var, size=I)

# Longitudinal model mean
mean_long = beta0 + X1*beta1

# Longitudinal model latent variables (eta_long)
eta_long = mean_long + ranef

# Survival submodel coefficients
if opt == "none" or opt == "nonph":
    gamma = -4  # Survival submodel coefficient for X1 (using X1 from longitudinal model)
    alpha = 0.2 # Longitudinal effect for survival submodel (can be adjusted)
    
    # Survival submodel linear predictor using X1 from longitudinal submodel
    eta_surv = X1 * gamma + eta_long * alpha
    
# Simulate Survival Times using Inverse Sampling Transform
scale = np.exp(8.5)
U = np.random.uniform(size=I)
alpha_beta = alpha * betat  # Product of alpha and betat
print(alpha_beta)
# Fix: Ensure CHF returns scalar values
def CHF(tau, i):
    def h(t,i):
        # Ensure that h(t) returns a scalar for each t
        if opt == "none" or opt == "interaction":
            return scale * np.exp(eta_surv[i] + alpha_beta * t)
        if opt == "nonph":
            return scale * np.exp(eta_surv[i] + 3 * X1[i] * np.sin(t) + alpha_beta * t)
    return np.exp(-1 * integrate.quad(lambda xi: h(xi,i), 0, tau)[0])
    
Ti = np.empty(I)
Ti[:] = np.NaN
for i in range(0, I):
    Ti[i] = optimize.brentq(lambda xi: U[i] - CHF(xi, i), 0, 100)

# Get true survival probabilities
true_prob = np.ones((I, len(obstime)))
for i in range(0, I):
    for j in range(1, len(obstime)):
        tau = obstime[j]
        true_prob[i, j] = CHF(tau, i)

C = np.random.uniform(low=obstime[3], high=obstime[-1] + 25, size=I)
C = np.minimum(C, obstime[-1])
event = Ti < C
#print(np.sum(event))
true_time = np.minimum(Ti, C)
time = [np.min([obs for obs in obstime if obs-t>=0]) for t in true_time]
#print(time)
J = len(obstime) 
subj_obstime = np.tile(obstime, reps=I)
pred_time = np.tile(obstime, reps=I)
mean_long = np.repeat(mean_long, repeats=J, axis=0)
eta_long = np.repeat(eta_long, repeats=J, axis=0)
long_err = np.random.normal(0, 1, size=I*J)
Y = eta_long + betat * subj_obstime + long_err#[:, np.newaxis]
Y_pred = eta_long + betat * pred_time + long_err#[:, np.newaxis]
true_prob = true_prob.flatten()
ID = np.repeat(range(0, I), repeats=J)
visit = np.tile(range(0, J), reps=I)
print(X1)
print(Y_pred.shape)
data = pd.DataFrame({"id":ID, "visit":visit, "obstime":subj_obstime, "predtime":pred_time,
                    "time":np.repeat(time,repeats=J),
                    "event":np.repeat(event,repeats=J),
                    "Y":Y,"X1":np.repeat(X1,repeats=J),
                    "pred_Y":Y_pred,"true":true_prob})
print(data.head(10))
