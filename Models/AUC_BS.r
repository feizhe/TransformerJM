library(survival)
library(tdROC)


AUC_1 = function(surv, event, time, pt){
    roc = tdROC( X = surv, Y = time, 
                 delta = event,
                 tau = pt, span = 0.05,
                 nboot = 0, alpha = 0.05,
                 n.grid = 1000, cut.off = 0.5)
    
    return(roc$main_res$AUC.integral)
}


AUC = function(surv, event, time, predtimes){
    aucs = rep(NA,length(predtimes))
    
    # if nonPH, surv should be matrix of Ixlength(predtimes)
    if(is.matrix(surv)){
        for(i in 1:length(predtimes)){
            print(i)
            print(surv[,i])
                  # Check if time is a numeric vector
            if (!is.numeric(time)) {
                time <- as.numeric(time)
            }
            
            # Check if X is a numeric vector (replace column 1 if needed)
            X <- surv[, i]
            if (!is.numeric(X)) {
                X <- as.numeric(X)
            }
            print(predtimes[i])
            aucs[i] = AUC_1(1-surv[,i], event, time, predtimes[i]) #####
        }
    }else if(is.vector(surv)){
        for(i in 1:length(predtimes)){
            print(i)
            print(surv)
                  # Check if time is a numeric vector
            if (!is.numeric(time)) {
                time <- as.numeric(time)
            }
            
            # Check if X is a numeric vector (replace column 1 if needed)
            X <- surv
            if (!is.numeric(X)) {
                X <- as.numeric(X)
            }
            print(predtimes[i])
            aucs[i] = AUC_1(1-X, event, time, predtimes[i])
        }
    }
    
    return(list("auc"=aucs))
}


Brier = function(surv, event, time, event_train, time_train, LT, DeltaT){
    #estimate km curve for BS calculation
    train.surv = cbind.data.frame("event"=event_train, "time"=time_train)
    km = survfit(Surv(time, event)~1, data=train.surv)
    survest = stepfun(km$time, c(1, km$surv))
    
    BS = rep(NA, length(DeltaT))
    for(i in 1:length(DeltaT)){
        pt = LT + DeltaT[i]
        N_vali = length(event)
        
        #BRIER SCORE
        D = rep(0, N_vali) 
        D[time<=pt & event==1] = 1
        
        pi = 1-surv[,i]
        
        km_pts = survest(time)/survest(LT)
        W2 <- D/km_pts
        W1 <- as.numeric(time>pt)/(survest(pt)/survest(LT))
        W <- W1 + W2
        
        BS_pts <- W * (D - pi)^2
        BS[i] = sum(na.omit(BS_pts)) / N_vali
    }
    return(BS)
}




surv_test <- matrix(runif(282*3),282,3)
e_tmp_test <- rep(1,282)
t_tmp_test <- c(7, 7, 4, 6, 7, 6, 4, 10, 10, 4, 6, 6, 7, 9, 2, 10, 5, 6, 6, 10, 10, 6, 10, 7,
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
                9, 2, 10, 2, 10, 6, 10, 10, 4, 8, 10, 10, 8, 3, 8, 10, 8, 10)
predtimes <- c(2,3,4)
roc <- tdROC( X = 1-surv_test[,1], Y = t_tmp_test, 
                 delta = e_tmp_test,
                 tau = predtimes, span = 0.05,
                 nboot = 0, alpha = 0.05,
                 n.grid = 1000, cut.off = 0.5)
roc$main_res$AUC.integral
#test <- AUC(1-surv_test, e_tmp_test, t_tmp_test, predtimes)


