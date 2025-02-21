import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats



def simulate_JM_base_1(I, obstime, miss_rate=0.1, opt="none", seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    J = len(obstime)
    
    #### longitudinal submodel ####
    beta0 = 1.5 # np.array([1.5,2,0.5],)
    beta1 = 2 # np.array([2,-1,1])
    betat = 1.5 # np.array([1.5, -1, 0.6]) 
    b_var = 1 # np.array([1,1.5,2])
    e_var = 1 # np.array([1,1,1])
    # rho = np.array([-0.2,0.1,-0.3])
    # b_Sigma = np.diag(b_var)
    # b_Sigma[0,1] = b_Sigma[1,0] = np.sqrt(b_var[0]*b_var[1])*rho[0]
    # b_Sigma[0,2] = b_Sigma[2,0] = np.sqrt(b_var[0]*b_var[2])*rho[1]
    # b_Sigma[1,2] = b_Sigma[2,1] = np.sqrt(b_var[1]*b_var[2])*rho[2]
    

    x1 = np.random.normal(3,1,size=I)
    ranef = np.random.normal(loc=0, scale = b_var, size=I)
    mean_long = beta0 + x1*beta1
    eta_long = mean_long[:,np.newaxis] + ranef[:,np.newaxis]


    if opt=="none" or opt=="nonph":
        gamma = 4 # np.array([-4,-2])
        alpha = 0.2 # np.array([0.2,-0.2,0.4])
        # x1 = np.random.binomial(n=1,p=0.5,size=I)
        # x2 = np.random.normal(loc = 1, scale=0.5 ,size=I)
        # W = np.stack((x1,x2), axis=1)
        eta_surv = x1*gamma + eta_long*alpha
        base = x1[:,np.newaxis]#W[...,np.newaxis]
        
    # if opt=="interaction":
        # alpha = np.array([0.2,-0.2,0.4])

        # x1 = np.random.binomial(n=1,p=0.5,size=I)

        # x2 = np.random.normal(size=I)

        # x3 = x1*x2

        # W = np.stack((x1,x2,x3), axis=1)

        # eta_surv = W@gamma + eta_long@alpha

        # base = np.stack((x1,x2), axis=1)

        # base = base[...,np.newaxis]
        # base = np.stack((x1,x2), axis=1)
        # base = base[...,np.newaxis]


    #Simulate Survival Times using Inverse Sampling Transform
    # phi = 1.5 # 
    scale= np.exp(-7)
    U = np.random.uniform(size=I)   
    alpha_beta = alpha*betat
    
    def CHF(tau,i):
        def h(t,i):
            if opt=="none" or opt=="interaction":
                return scale * np.exp(eta_surv[i] + alpha_beta*t)
            # (scale*(t**(scale-1))) * np.exp(eta_surv[i] + alpha_beta*t)
            # if opt=="nonph":
            #     return scale * np.exp(eta_surv[i] + 3*x1[i]*np.sin(t) + alpha_beta*t)
            #(scale*(t**(scale-1))) * np.exp(eta_surv[i] + 3*x2[i]*np.sin(t) + alpha_beta*t)
        return np.exp(-1 * integrate.quad(lambda xi: h(xi),0,tau)[0])
    # Inverse survival function
    # def invS(t, u, i):
    #     def h(s):
    #         Mm = alpha * eta_long[i] + alpha_beta * s
    #         return np.exp(np.log(phi) + (phi - 1) * np.log(s) + x1[i] * gamma + Mm)    
    #     result, _ = integrate.quad(h, 0, t)
    #     return result + np.log(u)    
 
 
    # u = np.random.uniform(size=I)
    # true_times = np.full(I, np.nan)

    # for i in range(I):
    #     up = 51
    #     tries = 45
    #     while tries > 0:
    #         try:
    #             root = optimize.brentq(lambda t: invS(t, u[i], i) - np.log(u[i]), 0, up)
    #             true_times[i] = root
    #             break
    #         except ValueError:
    #             tries -= 1
    #             up += 2
 
    Ti = np.empty(I)
    Ti[:] = np.NaN
    for i in range(0,I):
        Ti[i] = optimize.brentq(lambda xi: U[i]-CHF(xi,i), 0, 100)
    
    
    #Get true survival probabilities
    true_prob = np.ones((I, len(obstime)))
    for i in range(0,I):
        for j in range(1,len(obstime)):
            tau = obstime[j]
            true_prob[i,j] = CHF(tau)

    C = np.random.uniform(low=obstime[3], high=obstime[-1]+25, size=I)
    C = np.minimum(C, obstime[-1])
    event = Ti<C
    true_time = np.minimum(Ti, C)

    # round true_time up to nearest obstime
    time = [np.min([obs for obs in obstime if obs-t>=0]) for t in true_time]
    
    
    subj_obstime = np.tile(obstime, reps=I)
    pred_time = np.tile(obstime, reps=I)
    mean_long = np.repeat(mean_long, repeats=J, axis=0)
    eta_long = np.repeat(eta_long, repeats=J, axis=0)
    long_err = np.random.normal(0, 1, size=I*J)
    Y = eta_long + betat * subj_obstime + long_err[:, np.newaxis]
    Y_pred = eta_long + betat * pred_time + long_err[:, np.newaxis]
    true_prob = true_prob.flatten()
    ID = np.repeat(range(0, I), repeats=J)
    visit = np.tile(range(0, J), reps=I)
    print(x1)
    print(Y_pred.shape)
    data = pd.DataFrame({"id":ID, "visit":visit, "obstime":subj_obstime, "predtime":pred_time,
                        "time":np.repeat(time,repeats=J),
                        "event":np.repeat(event,repeats=J),
                        "Y":Y,
                        "X1":np.repeat(base[:,0],repeats=J),
                        "pred_Y":Y_pred,"true":true_prob})
    
    
    return data
    
simulate_JM_base_1(10, [1,2,3,4,5,6,7,8,9,10],seed=1).head()