library(survival)
library(tdROC)
library(JMbayes2)
library(nlme)
surv_pred <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/surv_pred_1.csv",header=FALSE)
testdata <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/test_data1.csv")
traindata <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/train_data1.csv")
tmpdata <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/tmp_data1.csv")

temp.time <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/time_tmp1.csv",header=FALSE)[-1,]
temp.event <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/event_tmp1.csv",header=FALSE)[-1,]
train.time <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/time_train1.csv",header=FALSE)
event.time <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/event_train1.csv",header=FALSE)

#test <- AUC(1-surv_pred, temp.event, temp.time, predtimes)
X <- 1-surv_pred
roc<- tdROC(X = X[,1], Y = temp.time, delta = temp.event, tau = 5, span = 0.05,
            nboot = 0, alpha = 0.05, n.grid = 1000, cut.off = 0.5)
roc$AUC
