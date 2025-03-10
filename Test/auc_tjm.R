library(survival)
library(tdROC)
library(JMbayes2)
library(nlme)
surv_pred <- read.csv("C:/research/TJM/TransformerJM/surv_pred_1.csv",header=FALSE)
testdata <- read.csv("C:/research/TJM/TransformerJM/test_data1.csv")
traindata <- read.csv("C:/research/TJM/TransformerJM/train_data1.csv")
tmpdata <- read.csv("C:/research/TJM/TransformerJM/tmp_data1.csv")

temp.time <- read.csv("C:/research/TJM/TransformerJM/time_tmp1.csv",header=FALSE)[-1,]
temp.event <- read.csv("C:/research/TJM/TransformerJM/event_tmp1.csv",header=FALSE)[-1,]
train.time <- read.csv("C:/research/TJM/TransformerJM/time_train1.csv",header=FALSE)
event.time <- read.csv("C:/research/TJM/TransformerJM/event_train1.csv",header=FALSE)

#test <- AUC(1-surv_pred, temp.event, temp.time, predtimes)
X <- 1-surv_pred

tdROC(X = X[,1], Y = temp.time, delta = temp.event, tau = 5, span = 0.05,
            nboot = 0, alpha = 0.05, n.grid = 1000, cut.off = 0.5,type="Epanechnikov")$AUC$value

pt <- seq(1.5,5,by=0.5)
aucstjm <- numeric(8)
for (i in 1:8){
  roc<- tdROC(X = X[,i], Y = temp.time, delta = temp.event, tau = pt[i], span = 0.05,
              nboot = 0, alpha = 0.05, n.grid = 1000, cut.off = 0.5,type="Epanechnikov")
  aucstjm[i]<-roc$AUC$value
}
aucstjm

