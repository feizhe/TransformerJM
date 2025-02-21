library(survival)
library(tdROC)
library(JMbayes2)
library(nlme)
library(JMbayes)



surv_pred <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/surv_pred1.csv",header=FALSE)
testdata <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/test_data1.csv")
traindata <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/train_data1.csv")
tmpdata <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/tmp_data1.csv")

temp.time <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/time_tmp1.csv",header=FALSE)[-1,]
temp.event <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/event_tmp1.csv",header=FALSE)[-1,]
train.time <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/time_train1.csv",header=FALSE)
event.time <- read.csv("C:/Users/jgmea/research/transf/TransformerJM/Test/event_train1.csv",header=FALSE)



traindata$event<-ifelse(traindata$event == "True", 1, 0)
traindata$obstime<-as.numeric(traindata$obstime)
traindata$id <- as.factor(traindata$id)

trd <- traindata[!duplicated(traindata$id), ]
common_ids <- intersect(unique(traindata$id), unique(trd$id))

# Subset both datasets to only include common IDs
traindata <- traindata[traindata$id %in% common_ids, ]
trd <- trd[trd$id %in% common_ids, ]


long <- lme(Y ~ X1 + obstime, data = na.omit(traindata),
            random = ~ 1| id)


cox.1 <- coxph(Surv(time,as.numeric(event))~X1,data=na.omit(trd),x=TRUE)


# Check for NaN or Inf in the longitudinal data
sum(is.nan(long$data$Y))  # Check NaN in Y
sum(is.nan(long$data$X1))  # Check NaN in X1
sum(is.nan(long$data$obstime))  # Check NaN in obstime

sum(is.infinite(long$data$Y))  # Check Inf in Y
sum(is.infinite(long$data$X1))  # Check Inf in X1
sum(is.infinite(long$data$obstime))  # Check Inf in obstime

# Check for NaN or Inf in the Cox model data (trd)
sum(is.nan(cox.1$data$time))  # Check NaN in time
sum(is.nan(cox.1$data$event))  # Check NaN in event
sum(is.infinite(cox.1$data$time))  # Check Inf in time
sum(is.infinite(cox.1$data$event))  # Check Inf in event

# Remove rows with NaN or Inf values in long
#long_clean <- long[!is.nan(long$Y) & !is.infinite(long$Y), ]
#long_clean <- long_clean[!is.nan(long_clean$X1) & !is.infinite(long_clean$X1), ]
#long_clean <- long_clean[!is.nan(long_clean$obstime) & !is.infinite(long_clean$obstime), ]

# Remove rows with NaN or Inf values in the Cox data
#cox.1_clean <- cox.1$data[!is.nan(cox.1$data$time) & !is.infinite(cox.1$data$time), ]
#cox.1_clean <- cox.1_clean[!is.nan(cox.1_clean$event) & !is.infinite(cox.1_clean$event), ]

jmfit <- jointModelBayes(long,cox.1,"obstime")
#jmtr<- jm(cox.1,long, time_var = "obstime", n_chains = 1L)


mcs <- as.mcmc(jmfit$mcmc)

plot(mcs$betas[,1],type="l",ylab="Intercept")
plot(mcs$betas[,2],type="l",ylab="X1")
plot(mcs$betas[,3],type="l",ylab="obstime")

plot(mcs$sigma,type="l",ylab="Sigma")

plot(mcs$alphas,type="l",ylab="alpha");abline(h=mean(mcs$alphas),col="blue",lwd=2)
mean(mcs$alphas)
plot(mcs$gammas,type="l",ylab="alpha")
traceplot(mcs$betas)
library(JMbayes)
