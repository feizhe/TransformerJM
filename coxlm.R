library(survival);library(dplyr)



fit_landmark_cox_with_Y <- function(df, landmark_time, horizon_s) {
  # Step 1: Select patients at risk at landmark time
  suppressMessages({
    survival_df <- df %>%
      group_by(id) %>%
      slice(1) %>%
      filter(time > landmark_time)
  })
  
  risk_ids <- survival_df$id
  
  # Step 2: Get most recent Y observation before landmark time
  Y_summary <- df %>%
    filter(id %in% risk_ids, obstime < landmark_time) %>%
    group_by(id) %>%
    arrange(obstime) %>%
    slice_tail(n = 1) %>%
    ungroup() %>%
    transmute(id, last_Y = Y)
  # Step 3: Merge with baseline and survival info
  design_df <- survival_df %>%
    dplyr::select(id, time, event, starts_with("X")) %>%
    left_join(Y_summary, by = "id") %>%
    mutate(
      new_time = pmin(time, landmark_time + horizon_s),
      new_event = ifelse(time <= landmark_time + horizon_s & event == 1, 1, 0),
      time_to_event = new_time - landmark_time
    )
  
  # Drop rows with missing Y (i.e., no observation before t)
  # design_df <- design_df %>% filter(!is.na(last_Y))
  
  # Step 4: Fit Cox model
  surv_obj <- Surv(design_df$time_to_event, design_df$new_event)
  covariates <- suppressMessages({design_df %>% dplyr::select(starts_with("X"), last_Y)}) 
  
  model <- coxph(surv_obj ~ .-id, data = covariates)
  
  return(model)
}


unq <- unique(data$id)

trsb <- sample(unq, size = 0.7*length(unq))

trd <- data[data$id %in% trsb, ];tsd <- data[!data$id %in% trsb, ]

ph1 <- suppressMessages(fit_landmark_cox_with_Y(trd,1,0.5)) 




pr1 <- predict_landmark_survival(ph1,tsd,1,0.5)

pr1$surv_prob[1,]


pred_cox_lm <- function(model,new_data,lt,s){
  
  # Step 1: Prepare test subjects at risk at t
  survival_df <- new_data %>%
    group_by(id) %>%
    slice(1) %>%
    filter(time > lt)  # t = 1
  
  risk_ids <- survival_df$id
  
  # Step 2: Get latest Y up to t
  Y_summary <- new_data %>%
    filter(id %in% risk_ids, obstime <= lt) %>%
    group_by(id) %>%
    slice_max(order_by = obstime, n = 1) %>%
    dplyr::select(id, last_Y = Y)
  
  # Step 3: Merge with covariates
  design_df <- survival_df %>%
    dplyr::select(id, starts_with("X")) %>%
    left_join(Y_summary, by = "id") %>%
    filter(!is.na(last_Y))
  
  # Step 4: Predict survival curve
  sf <- survfit(model, newdata = design_df)
  
  # Step 5: Extract survival probs at s = 0.5 (i.e., time = 1.5)
  sf_summary <- summary(sf, times = s)  # Note: times is relative to landmark time
  
  # Step 6: Get predicted probabilities
  surv_probs <- t(sf_summary$surv)  # Vector of P(T > 1.5 | T > 1) per subject
  
  # Step 7: Combine with subject IDs
  preds <- data.frame(id = survival_df$id, surv_prob = surv_probs, time = survival_df$time,
                      event=survival_df$event)
  return(preds)
}


#ps1 <- pred_cox_lm(ph1,tsd,1,0.5)

library(tdROC)


#tdROC(1-ps1$surv_prob,ps1$time,ps1$event,1.5)



#Brier(ps1$surv_prob,ps1$event,ps1$time,trd$event,trd$time,1,.5)




ph_auc <- matrix(nrow = 180, ncol = 5)  # Adding column for dataset ID
colnames(ph.auc.lin) <- c("dataset", "landmark_time", "prediction_window", "AUC","BR")


lt <- c(1, 1.5, 2)
pw <- c(0.5, 1)
rc <- 1
options(warn=1)
dst <- 1:30
ph.auc.lin <- matrix(nrow=180,ncol=5)
for (k in dst) {
  for (l in 1:3) {
    for (w in 1:2) {
      t <- lt[l]
      s <- pw[w]
      path <- paste("C:/Users/jgmea/OneDrive/Desktop/datasets/lin10/r_data_lin10_",k,".csv",sep="")
      
      the_data <- read.csv(path)
      
      
      unq <- unique(the_data$id)
      
      trsb <- sample(unq, size = 0.7*length(unq))
      
      trd <- the_data[the_data$id %in% trsb, ];tsd <- the_data[!the_data$id %in% trsb, ]
      
      phfit <- suppressMessages(fit_landmark_cox_with_Y(trd,t,s))
      
      
      ps <- pred_cox_lm(phfit,tsd,t,s)
      
      
      
      ph.auc <- tdROC(1-ps$surv_prob,ps$time,ps$event,t+s,span=0.05, type="Epachinikov")
      
      
      
      BS<- round(Brier(ps$surv_prob,ps$event,ps$time,trd$event,trd$time,t,s),4)
      ph.auc.lin[rc,]<- c(k, t, s, ph.auc$AUC$value,BS)
      rc<-rc+1
      
    }
  }
}

matl <- matrix(nrow=6,ncol=6)
colnames(matl)<-c("AUC","AUC.sd","AUC.q1","AUC.q2","BS","BS.sd")
rl<-1
for (l in 1:3) {
  for (w in 1:2) {
    t <- lt[l]
    s <- pw[w]
    merp <- subset(as.data.frame(ph.auc.lin), landmark_time == t & prediction_window==s)
    # cat("Mean AUC for time", t, "at time ", t+s,":",mean(merp$AUC), "\n")
    # cat("SD AUC for time", t, "at time ", t+s,":",sd(merp$AUC), "\n")
    # cat("2.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.025), "\n")
    # cat("97.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.975), "\n")  
    # 
    # cat("Mean Brier Score for time", t, "at time ", t+s,":",mean(merp$BR), "\n")
    # cat("SD Brier Score for time", t, "at time ", t+s,":",sd(merp$BR), "\n")
    # cat("2.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs = 0.025), "\n")
    # cat("97.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs=0.975), "\n") 
    # 
    # 
    matl[rl,]<- c(mean(merp$AUC),sd(merp$AUC),quantile(merp$AUC,probs=c(0.025,0.975)),mean(merp$BR),
                  sd(merp$BR))
    rl <- rl+1
    
    
  }
}



round(matl,4)



##### NL terms





ph.auc.nl <- matrix(nrow=180,ncol=5) # Adding column for dataset ID
colnames(ph.auc.nl) <- c("dataset", "landmark_time", "prediction_window", "AUC","BR")


lt <- c(1, 1.5, 2)
pw <- c(0.5, 1)
rc <- 1
options(warn=1)
dst <- 1:30

for (k in dst) {
  for (l in 1:3) {
    for (w in 1:2) {
      t <- lt[l]
      s <- pw[w]
      path <- paste("C:/Users/jgmea/OneDrive/Desktop/datasets/nl10/r_data_nl10_",k,".csv",sep="")
      
      the_data <- read.csv(path)
      
      
      unq <- unique(the_data$id)
      
      trsb <- sample(unq, size = 0.7*length(unq))
      
      trd <- the_data[the_data$id %in% trsb, ];tsd <- the_data[!the_data$id %in% trsb, ]
      
      phfit <- suppressMessages(fit_landmark_cox_with_Y(trd,t,s))
      
      
      ps <- pred_cox_lm(phfit,tsd,t,s)
      
      
      
      ph.auc <- tdROC(1-ps$surv_prob,ps$time,ps$event,t+s,span=0.05, type="Epachinikov")
      
      
      
      BS<- round(Brier(ps$surv_prob,ps$event,ps$time,trd$event,trd$time,t,s),4)
      ph.auc.nl[rc,]<- c(k, t, s, ph.auc$AUC$value,BS)
      rc<-rc+1
      
    }
  }
}

matnl <- matrix(nrow=6,ncol=6)
colnames(matnl)<-c("AUC","AUC.sd","AUC.q1","AUC.q2","BS","BS.sd")
rnl<-1

for (l in 1:3) {
  for (w in 1:2) {
    t <- lt[l]
    s <- pw[w]
    merp <- subset(as.data.frame(ph.auc.nl), landmark_time == t & prediction_window==s)
    # cat("Mean AUC for time", t, "at time ", t+s,":",mean(merp$AUC), "\n")
    # cat("SD AUC for time", t, "at time ", t+s,":",sd(merp$AUC), "\n")
    # cat("2.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.025), "\n")
    # cat("97.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.975), "\n")  
    # 
    # cat("Mean Brier Score for time", t, "at time ", t+s,":",mean(merp$BR), "\n")
    # cat("SD Brier Score for time", t, "at time ", t+s,":",sd(merp$BR), "\n")
    # cat("2.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs = 0.025), "\n")
    # cat("97.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs=0.975), "\n") 
    # 
    # 
    matnl[rnl,]<- c(mean(merp$AUC),sd(merp$AUC),quantile(merp$AUC,probs=c(0.025,0.975)),mean(merp$BR),
                    sd(merp$BR))
    rnl <- rnl+1
  }
}

round(matnl,4)


### 70% censoring




ph_auc70 <- matrix(nrow = 180, ncol = 5)  # Adding column for dataset ID
colnames(ph_auc70) <- c("dataset", "landmark_time", "prediction_window", "AUC","BR")


lt <- c(1, 1.5, 2)
pw <- c(0.5, 1)
rc <- 1
options(warn=1)
dst <- 1:30

for (k in dst) {
  for (l in 1:3) {
    for (w in 1:2) {
      t <- lt[l]
      s <- pw[w]
      path <- paste("C:/Users/jgmea/OneDrive/Desktop/datasets/lin10/r_data_lin70_",k,".csv",sep="")
      
      the_data <- read.csv(path)
      
      
      unq <- unique(the_data$id)
      
      trsb <- sample(unq, size = 0.7*length(unq))
      
      trd <- the_data[the_data$id %in% trsb, ];tsd <- the_data[!the_data$id %in% trsb, ]
      
      phfit <- suppressMessages(fit_landmark_cox_with_Y(trd,t,s))
      
      
      ps <- pred_cox_lm(phfit,tsd,t,s)
      
      
      
      ph.auc <- tdROC(1-ps$surv_prob,ps$time,ps$event,t+s,span=0.05, type="Epachinikov")
      
      
      
      BS<- round(Brier(ps$surv_prob,ps$event,ps$time,trd$event,trd$time,t,s),4)
      ph_auc70[rc,]<- c(k, t, s, ph.auc$AUC$value,BS)
      rc<-rc+1
      
    }
  }
}

mat70 <- matrix(nrow=6,ncol=6)
colnames(mat70)<-c("AUC","AUC.sd","AUC.q1","AUC.q2","BS","BS.sd")
rc70 <- 1
for (l in 1:3) {
  for (w in 1:2) {
    t <- lt[l]
    s <- pw[w]
    merp <- subset(as.data.frame(ph_auc70), landmark_time == t & prediction_window==s)
    # cat("Mean AUC for time", t, "at time ", t+s,":",mean(merp$AUC), "\n")
    # cat("SD AUC for time", t, "at time ", t+s,":",sd(merp$AUC), "\n")
    # cat("2.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.025), "\n")
    # cat("97.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.975), "\n")  
    # 
    # cat("Mean Brier Score for time", t, "at time ", t+s,":",mean(merp$BR), "\n")
    # cat("SD Brier Score for time", t, "at time ", t+s,":",sd(merp$BR), "\n")
    # cat("2.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs = 0.025), "\n")
    # cat("97.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs=0.975), "\n") 
    # 
    # 
    
    
    mat70[rc70,]<- c(mean(merp$AUC,na.rm=T),sd(merp$AUC,na.rm=T),quantile(merp$AUC,probs=c(0.025,0.975),na.rm=T),mean(merp$BR,na.rm=T),
                     sd(merp$BR,na.rm=T))
    rc70 <- rc70+1
    
  }
}
round(mat70,3)





#### ar(1)



ph_aucar <- matrix(nrow = 180, ncol = 5)  # Adding column for dataset ID
colnames(ph_aucar) <- c("dataset", "landmark_time", "prediction_window", "AUC","BR")


lt <- c(1, 1.5, 2)
pw <- c(0.5, 1)
rc <- 1
options(warn=1)
dst <- 1:30

for (k in dst) {
  for (l in 1:3) {
    for (w in 1:2) {
      t <- lt[l]
      s <- pw[w]
      path <- paste("C:/Users/jgmea/OneDrive/Desktop/datasets/lin10/r_data_ar_",k,".csv",sep="")
      
      the_data <- read.csv(path)
      
      
      unq <- unique(the_data$id)
      
      trsb <- sample(unq, size = 0.7*length(unq))
      
      trd <- the_data[the_data$id %in% trsb, ];tsd <- the_data[!the_data$id %in% trsb, ]
      
      phfit <- suppressMessages(fit_landmark_cox_with_Y(trd,t,s))
      
      
      ps <- pred_cox_lm(phfit,tsd,t,s)
      
      
      
      ph.auc <- tdROC(1-ps$surv_prob,ps$time,ps$event,t+s,span=0.05, type="Epachinikov")
      
      
      
      BS<- round(Brier(ps$surv_prob,ps$event,ps$time,trd$event,trd$time,t,s),4)
      ph_aucar[rc,]<- c(k, t, s, ph.auc$AUC$value,BS)
      rc<-rc+1
      
    }
  }
}

matar <- matrix(nrow=6,ncol=6)
colnames(matar)<-c("AUC","AUC.sd","AUC.q1","AUC.q2","BS","BS.sd")
rcar <- 1
for (l in 1:3) {
  for (w in 1:2) {
    t <- lt[l]
    s <- pw[w]
    merp <- subset(as.data.frame(ph_aucar), landmark_time == t & prediction_window==s)
    # cat("Mean AUC for time", t, "at time ", t+s,":",mean(merp$AUC), "\n")
    # cat("SD AUC for time", t, "at time ", t+s,":",sd(merp$AUC), "\n")
    # cat("2.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.025), "\n")
    # cat("97.5 quantile AUC for time", t, "at time ", t+s,":",quantile(merp$AUC,probs=0.975), "\n")  
    # 
    # cat("Mean Brier Score for time", t, "at time ", t+s,":",mean(merp$BR), "\n")
    # cat("SD Brier Score for time", t, "at time ", t+s,":",sd(merp$BR), "\n")
    # cat("2.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs = 0.025), "\n")
    # cat("97.5 quantile Brier Score for time", t, "at time ", t+s,":",quantile(merp$BR,probs=0.975), "\n") 
    # 
    # 
    
    
    matar[rcar,]<- c(mean(merp$AUC),sd(merp$AUC),quantile(merp$AUC,probs=c(0.025,0.975)),mean(merp$BR),
                     sd(merp$BR))
    rcar <- rcar+1
    
  }
}
round(matar,3)


