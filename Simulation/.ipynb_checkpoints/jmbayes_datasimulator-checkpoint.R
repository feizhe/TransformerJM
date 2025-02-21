library(nlme)
library(survival)
library(JMbayes)



obstime <- seq(0,10,length=21)
I<-5000#;opt="none"

# simulate_JM_base2 <- function(I,  obstime, miss_rate = 0.1, opt = "none", seed = NULL) {
#   if (!is.null(seed)) {
#     set.seed(seed)
#   }

  
  J <- length(obstime)

  dat <- data.frame(id = rep(1:I, each = J))
  dat$visit <- rep(0:(J - 1), I)
  dat$obstime = rep(obstime,  I) 
  dat$predtime = rep(obstime, I)
  ### Longitudinal submodel ###
  beta0 <- -2.5
  beta1 <- 2
  betat <- 1.5
  b_var <- 2.5
  e_var <- 1
  #rho <- -0.2
  
  ## Error term -- time-varying, one per measurement time
  long_err  <- with(dat, rnorm(I*J, sd = sqrt(e_var)))
  
  # Covariate X1 (same for both submodels)
  X1 <- rnorm(I, mean = 0, sd = 1)
  dat$X1 <- rep(X1, each = J) ## not time-varying
  
  # Random effects for longitudinal model
  ranef <- rnorm(I, mean = 0, sd = sqrt(b_var))
  dat$subj.random <- rep(ranef, each = J)
  
  ##longitudinal observation
  dat$Y <- with(dat, beta0  + beta1*X1 + betat*obstime + subj.random + long_err)
  
  #predicted longitudinal observation
  pred_time <- rep(obstime, I)
  dat$pred_Y <- with(dat, beta0  + beta1*X1 + betat*pred_time + subj.random + long_err)
  
  # Survival submodel coefficients
  #if (opt == "none" || opt == "nonph") {
   
    gamma <- 1.5  # Survival submodel coefficient for X1 (using X1 from longitudinal model)
    alpha <- 0.9   # Longitudinal effect for survival submodel (can be adjusted)
    
    # Survival submodel linear predictor using X1 from longitudinal submodel
  #  eta_surv <- X1 * gamma + eta_long * alpha
  #}
  
  # Simulate Survival Times using Inverse Sampling Transform
  phi <- 3
  U <- runif(I)
 # alpha_beta <- alpha * betat  # Product of alpha and betat
  
  ## I am not sure why we are exponentiating again. 
  
  # # Hazard function (CHF)
  CHF <- function(tau, i) {
    h <- function(t, i) {
      #if (opt == "none" || opt == "interaction") {
        Mm = beta0 + X1[i]*beta1 + t * betat + ranef[i]
        return(exp(log(phi) + (phi - 1) * log(t) +  X1[i] * gamma + alpha*Mm))
      #}
      # if (opt == "nonph") {
      #   return(exp(log(phi) + (phi - 1) * log(t)) * exp(eta_surv[i] + 3 * X1[i] * sin(t) + alpha_beta * t))
      #}
    }
    return(exp(-integrate(function(xi) h(xi, i), 0, tau)$value))
  }
# 
#   Ti <- rep(NA, I)
#   for (i in 1:I) {
#     Ti[i] <- uniroot(function(xi) U[i] - CHF(xi, i), c(0, 100))$root
#   }
  
  invS = function (t, u, i) 
  {
    h = function(s) 
    {
      Mm = beta0 + X1[i]*beta1 + s * betat + ranef[i] 
      exp(log(phi) + (phi - 1) * log(s) + X1[i] * gamma + alpha*Mm)
    }     
    
    integrate(h, lower = 0, upper = t, subdivisions = 2000)$value + log(u)			
  }
  
  trueTimes = numeric(I)
  i = 1
  
  while(i<=I) {
    Up <- 51
    tries <- 45
    #print(i)
    Root <- try(uniroot(invS, interval = c(0, Up), u = U[i], i = i)$root, TRUE)
    while(inherits(Root, "try-error") && tries > 0) {
      tries <- tries - 1
      Up <- Up + 2
      Root <- try(uniroot(invS, interval = c(0, Up), u = U[i], i = i)$root, TRUE)
    }
    trueTimes[i] <- if (!inherits(Root, "try-error")) Root else NA
    if(is.na(trueTimes[i])==TRUE)
      i = i 
    else 
      i = i + 1
  }
  
  mean(trueTimes);hist(trueTimes)
  
  # Get true survival probabilities
  true_prob <- matrix(1, nrow = I, ncol = length(obstime))
  for (i in 1:I) {
    for (j in 2:length(obstime)) {
      tau <- obstime[j]
      true_prob[i, j] <- CHF(tau, i)
    }
  }
  
  C <- rexp(I, 0.15)
  C <- pmin(C, obstime[length(obstime)])
  event <- as.numeric(trueTimes <= C)#;sum(event)
  event.time <- pmin(trueTimes, C)
  1-sum(event)/I #censoring rate
  # Continuous version of time
  #ctstime <- event.time
  
  dat$event = rep(event, each = J)
  dat$time = rep(event.time, each = J)
  # Round true_time up to nearest obstime
  #time <- sapply(event.time, function(t) min(obstime[obstime - t >= 0]))
  
  
  true_prob <- as.vector(t(true_prob))
  
  dat$true <- true_prob
  #ID <- rep(0:(I - 1), each = J)
  visit <- rep(0:(J - 1), I)
  dat$r_time <- rep(sapply(event.time, function(t) min(obstime[obstime - t >= 0])),each=J)

  # data <- data.frame(
  #   id = ID, visit = visit, obstime = subj_obstime, predtime = pred_time,
  #   #time = rep(time, each = J), 
  #   event.time = rep(event.time, each = J),
  #   event = rep(event, each = J), Y = Y, X1 = rep(X1, each = J),
  #   pred_Y = Y_pred, true = true_prob#, true_time = true_time)
  data<- dat
  
  data<-data[data$obstime<=data$time,]
 # return(data)
#}

# Example usage
set.seed(2)
#data <- simulate_JM_base2(1000, seq(0,10,length=20))
# print(head(data,8))
# print(sum(data$event)/dim(data)[1])
# print(mean(data$ctstime))
# print(mean(data$Y))
# print(head(data,8))



data2 <- data[!duplicated(data$id),]

test_id <- 1:(round(0.7*I))

test_data <- data[data$id%in%test_id,]

test_data2 <- data2[data2$id%in%test_id,]

long2 <- lme(Y ~ X1 + obstime, data = data, random = ~ 1|id) 


cox.2 <- coxph(Surv(time, event)~X1, data = data2, x = TRUE)

jmfit2 <- jointModelBayes(long2, cox.2, timeVar = "obstime")


 summary(jmfit2)

traceplot(jmfit2$mcmc)


mcs <- jmfit2$mcmc

plot(mcs$betas[, 1], type = "l", ylab = "Intercept")
plot(mcs$betas[, 2], type = "l", ylab = "X1")
plot(mcs$betas[, 3], type = "l", ylab = "obstime")

plot(mcs$sigma, type = "l", ylab = "Sigma")


plot(mcs$alphas, type = "l", ylab = "alpha");abline(h=mean(mcs$alphas),col="blue",lwd=2)
mean(mcs$alphas)
plot(mcs$gammas, type = "l", ylab = "gamma")



r_data <- data[,c("id","visit","obstime","predtime","r_time","time", "event","Y","X1","pred_Y","true")]
names(r_data)[names(r_data) == 'time'] <- 'ctstime'
names(r_data)[names(r_data) == 'r_time'] <- 'time'
r_data$event <- ifelse(r_data$event==1,T,F)


write.csv(r_data,file="r_data.csv",row.names = FALSE)



test_id <- (round(0.7*I)+1):I

test_data <- data[data$id%in%test_id,]

LT <- 1

lts <- seq(2,6,by=0.5)

survPred <- survfitJM(jmfit2, newdata = test_data, idVar = "id", survTimes = lts)

#?ifelse



