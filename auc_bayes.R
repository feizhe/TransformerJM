library(nlme)
library(survival)
library(JMbayes)
library(tictoc)
library(basicMCMCplots)




pt <- seq(1,4.5,by=0.5)
censor.rate <- numeric(length(pt))
obsleft<- numeric(8)
aucs <- numeric(length(seq(1,4.5,by=0.5)))
aucs_hence <- numeric(8)

for (i in 1:length(pt)){
  chain_auc <- numeric(3)
  auch <- numeric(3)
  for (j in 1:3){
    chain_auc[j] <- aucJM(chains[[j]],newdata = test_data, Tstart=pt[i],Thoriz = 5)$auc
    auch[j] <- aucJM(chains[[j]],newdata = test_data, Tstart=1,Thoriz = pt[i]+0.5)$auc
  }
  aucs[i] <- mean(chain_auc)
  aucs_hence[i]<-mean(auch)
  censor.rate[i] <- 1-mean(subset(test_data, time > pt[i] & obstime==0)$event)
  obsleft[i] <- sum((test_data[test_data$visit==0,]$time > pt[i]))
}
censor.rate
obsleft

aucs;aucs_hence

aucJM(chains[[1]],newdata = test_data, Tstart=1,Thoriz = 5)$auc



aucJM(jmfit2,newdata = test_data, Tstart=1.5, Thoriz = 5)
aucJM(jmfit2,newdata = test_data, Tstart=2, Thoriz = 5)
aucJM(jmfit2,newdata = test_data,Tstart = 2.5,Thoriz = 5)
aucJM(jmfit2,newdata = test_data,Tstart = 3,Thoriz = 5)
aucJM(jmfit2,newdata = test_data,Tstart = 3.5,Thoriz = 5)
aucJM(jmfit2,newdata = test_data,Tstart = 4,Thoriz = 5)
aucJM(jmfit2,newdata = test_data,Tstart = 4.5,Thoriz = 5)
