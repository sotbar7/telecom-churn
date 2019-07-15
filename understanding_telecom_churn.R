#########################################
#   TELECOM CHURN PREDICTION PROJECT    #
#   SOTIRIS BARATSAS                    #
#   Contact: sotbaratsas[at]gmail.com   #
#########################################

library(glmnet)
library(aod)
library(psych)
library(car)

### DATA PREPARATION ###

dataset <- read.csv(file="churn2.csv", header=TRUE, sep=",")
View(dataset)
dim(dataset)
dataset <- dataset[,c('Churn', colnames(dataset)[colnames(dataset)!='Churn'])]
str(dataset)
dataset$Day.Mins <- as.numeric(dataset$Day.Mins)
dataset$Eve.Mins <- as.numeric(dataset$Eve.Mins)
dataset$Night.Mins <- as.numeric(dataset$Night.Mins)
dataset$Intl.Mins <- as.numeric(dataset$Intl.Mins)
dataset$Churn <- as.factor(dataset$Churn)
dataset$Int.l.Plan <- as.factor(dataset$Int.l.Plan)
dataset$VMail.Plan <- as.factor(dataset$VMail.Plan)
dataset$Area.Code <- as.factor(dataset$Area.Code)
dataset$Day.Charge <- as.numeric(dataset$Day.Charge)
dataset$Eve.Charge <- as.numeric(dataset$Eve.Charge)
dataset$Night.Charge <- as.numeric(dataset$Night.Charge)
dataset$Intl.Charge <- as.numeric(dataset$Intl.Charge)
str(dataset)

### EXPLORATORY DATA ANALYSIS ###

summary(dataset)

# Separating the numeric & integer columns, in order to analyse them better.
index <- sapply(dataset, class) != "factor"
numcolumns <- dataset[,index]
round(t(describe(numcolumns)),2)
sapply(numcolumns, sd)
# Separating the factor columns
fcolumns <- dataset[,!index] # for Factors
n <- nrow(dataset)
k = 20 # number of predictors


par(mfrow=c(2,4))
for(i in 1:8){
  hist(numcolumns[,i], xlab=names(numcolumns)[i], main=names(numcolumns)[i])
}
for(i in 9:15){
  hist(numcolumns[,i], xlab=names(numcolumns)[i], main=names(numcolumns)[i])
}
par(mfrow=c(2,4))
for(i in 1:8){
  qqnorm(numcolumns[,i], xlab=names(numcolumns)[i], main=names(numcolumns)[i])
}
for(i in 9:15){
  qqnorm(numcolumns[,i], xlab=names(numcolumns)[i], main=names(numcolumns)[i])
}


par(mfrow=c(1,1))
barplot(prop.table(table(fcolumns$Churn, fcolumns$State), 2)*100, col=c("seagreen3", "indianred3"), main="States", cex.names = 0.95, las=2)
par(mfrow=c(2,2))
barplot(prop.table(table(fcolumns$Churn, fcolumns$Gender), 2)*100, col=c("seagreen3", "indianred3"), main="Gender")
barplot(prop.table(table(fcolumns$Churn, fcolumns$Area.Code), 2)*100, col=c("seagreen3", "indianred3"), main="Area Code")
barplot(prop.table(table(fcolumns$Churn, fcolumns$VMail.Plan), 2)*100, col=c("seagreen3", "indianred3"), main="VMail Plan")
barplot(prop.table(table(fcolumns$Churn, fcolumns$Int.l.Plan), 2)*100, col=c("seagreen3", "indianred3"), main="Intl Plan")


### In case we want a legend that appears next to the barplot:
# par(mar=c(4,4,4,10), xpd=TRUE)
# barplot(prop.table(table(fcolumns$Churn, fcolumns$State), 2)*100, col=c("seagreen3", "indianred3"))
# legend("topleft", inset=c(1,0.05), legend = c("Stayed", "Left"), fill = c("seagreen3", "indianred3"), cex=0.75)
# dev.off() # to reset the margins

# If we want to see the absolute values instead of the proportions
# ggplot(data=fcolumns, aes(x=State, y=Churn, fill=Churn)) +
# geom_bar(aes(y = (..count..)/sum(..count..))) + 
#   scale_y_continuous(labels = scales::percent)



### EXAMINING CORRELATION BETWEEN VARIABLES

round(cor(numcolumns), 2)
library(corrplot)
par(mfrow=c(1,1))
corrplot(cor(numcolumns), method = "number", tl.cex=0.55) 

# par(mfrow=c(1,1))
# pairs(numcolumns)

### MODEL FORMULATION ###

attach(dataset)
model0 <- glm(Churn~., data = dataset, family = "binomial")
summary(model0)

### VARIABLE SELECTION ###

# STEP AIC

model1 <- step(model0, trace=TRUE, direction = 'both')
summary(model1)

library(car)
round(vif(model1),1) # We get a VIF value for each variable

# REMOVING THE VARIABLE VMail.Message
model2<- update(model1, . ~ .-VMail.Message)
summary(model2)
round(vif(model2),1)


# Using Cook's Distance to identify leverage points and possibly remove some observations to improve the model
cooksd <- cooks.distance(model2)
par(mfrow=c(1,1))
plot(cooksd, pch="*", cex=0.65, main="Influential Observations by Cook's distance")
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # adding cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4/(n-k-1),names(cooksd),""), col="red")

# What are we going to use as the cut-off point?
length(cooksd[cooksd > 1])
length(cooksd[cooksd > 4/(n)])
length(cooksd[cooksd > 4/(n-k-1)])
length(cooksd[cooksd > 4*mean(cooksd)])
length(cooksd[cooksd > quantile(cooksd, 0.99)])


#influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])  # influential row numbers
influential <- as.numeric(names(cooksd)[(cooksd > quantile(cooksd, 0.99))])  # influential row numbers
head(dataset[influential, ])
dataset <- dataset[-influential, ]
length(dataset[,1])

# Re-calculating all models.
model0 <- glm(Churn~., data = dataset, family = "binomial")
summary(model0)
round(vif(model0), 1)

# Individual Model Evaluation
logisticPseudoR2s <- function(LogModel) {
  dev <- LogModel$deviance
  nullDev <- LogModel$null.deviance
  modelN <- length(LogModel$fitted.values)
  R.l <-  1 -  dev / nullDev
  R.cs <- 1- exp ( -(nullDev - dev) / modelN)
  R.n <- R.cs / ( 1 - ( exp (-(nullDev / modelN))))
  R.hlp <- (ResourceSelection::hoslem.test(LogModel$y, fitted(LogModel), g=10))$p.value
  cat("Pseudo R^2 for logistic regression\n")
  cat("McFadden's R^2             ", round(R.l, 3), "\n")
  cat("Cox and Snell R^2          ", round(R.cs, 3), "\n")
  cat("Nagelkerke R^2             ", round(R.n, 3),    "\n")
  cat("Hosmer and Lemeshow p-value", round(R.hlp, 3),    "\n")
} 

logisticPseudoR2s(model0)

model1 <- step(model0, trace=TRUE, direction = 'both')
summary(model1)
round(vif(model1),1)

model2<- update(model1, . ~ .-VMail.Message)
logisticPseudoR2s(model2)

# STEP BIC

n <- nrow(dataset)
model3 <- step(model0, trace=TRUE, direction = 'both', k=log(n))
summary(model3)

round(vif(model3),1) # No multicollinearity problem this time
logisticPseudoR2s(model3)


# LASSO

modmtrx <- model.matrix(model0)[,-1]
lasso <- glmnet(modmtrx, dataset$Churn, alpha=1, family='binomial')
par(mfrow=c(1,1))
library(plotmo)
plot_glmnet(lasso)

lasso1 <- cv.glmnet(modmtrx, dataset$Churn, alpha = 1, family='binomial')
lasso1$lambda.min
lasso1$lambda.1se
coef(lasso1,s=lasso1$lambda.1se)
model_l <- lasso1
cv.out <- cv.glmnet(modmtrx,dataset$Churn,alpha=1,family="binomial",type.measure = "mse")
plot(cv.out)

lasso_prob <- predict(lasso1,newx = modmtrx,s=lasso1$lambda.1se,type="response")
#translate probabilities to predictions
lasso_predict <- rep(0,nrow(dataset))
lasso_predict[lasso_prob>.5] <- 1
#confusion matrix
table(pred=lasso_predict,true=dataset$Churn)
#accuracy
mean(lasso_predict==dataset$Churn)

### VARIABLE TRANSFORMATIONS ###
# Because a log or polynomial transformation would have negative consequences on our ability to interpret the model in a practical way, we will only perform some simple (aggregate) transformations

# Day.Mins + Eve.Mins + Night.Mins = Domestic.Mins

data4 <- dataset[, !names(dataset) %in% c("Day.Mins", "Eve.Mins", "Night.Mins")] 
data4$Domestic.Mins <- dataset$Day.Mins + dataset$Eve.Mins + dataset$Night.Mins

model4 <- glm(Churn~., data = data4, family = "binomial")
summary(model4)

model5 <- step(model4, trace=TRUE, direction = 'both')
summary(model5)
round(vif(model5),1)
model6<- update(model5, . ~ .-VMail.Message) # Removing the variable VMail.Message because the VIFs show it is highly collinear with VMail.Plan
summary(model6)
round(vif(model6),1)

logisticPseudoR2s(model6)

# Day.Calls + Eve.Calls + Night.Calls = Domestic.Calls

data7 <- dataset[, !names(dataset) %in% c("Day.Calls", "Eve.Calls", "Night.Calls")] 
data7$Domestic.Calls <- dataset$Day.Calls + dataset$Eve.Calls + dataset$Night.Calls

model7 <- glm(Churn~., data = data7, family = "binomial")
summary(model7)

model8 <- step(model7, trace=TRUE, direction = 'both')
summary(model8)
round(vif(model8),1)
model9<- update(model8, . ~ .-VMail.Message) # Removing the variable VMail.Message because the VIFs show it is highly collinear with VMail.Plan
summary(model9)
round(vif(model9),1)

logisticPseudoR2s(model9)

# Combination of both

data10 <- dataset[, !names(dataset) %in% c("Day.Calls", "Eve.Calls", "Night.Calls", "Day.Mins", "Eve.Mins", "Night.Mins")]
data10$Domestic.Mins <- dataset$Day.Mins + dataset$Eve.Mins + dataset$Night.Mins
data10$Domestic.Calls <- dataset$Day.Calls + dataset$Eve.Calls + dataset$Night.Calls

model10 <- glm(Churn~., data = data10, family = "binomial")
summary(model10)
model11 <- step(model10, trace=TRUE, direction = 'both')
summary(model11)
round(vif(model11),1)
model12<- update(model11, . ~ .-VMail.Message) # Removing the variable VMail.Message because the VIFs show it is highly collinear with VMail.Plan
summary(model12)
round(vif(model12),1)

logisticPseudoR2s(model12)

# Day.Charge + Eve.Charge + Night.Charge = Domestic.Charge

data13 <- dataset[, !names(dataset) %in% c("Day.Charge", "Eve.Charge", "Night.Charge")] 
data13$Domestic.Charge <- dataset$Day.Charge + dataset$Eve.Charge + dataset$Night.Charge

model13 <- glm(Churn~., data = data13, family = "binomial")
summary(model13)

model14 <- step(model13, trace=TRUE, direction = 'both')
summary(model14)
round(vif(model14),1)
model15<- update(model14, . ~ .-VMail.Message) # Removing the variable VMail.Message because the VIFs show it is highly collinear with VMail.Plan
summary(model15)
round(vif(model15),1)

logisticPseudoR2s(model15)

### MODEL EVALUATION: Choosing the best model ###

# Individual Model Evaluation

logisticPseudoR2s <- function(LogModel) {
  dev <- LogModel$deviance
  nullDev <- LogModel$null.deviance
  modelN <- length(LogModel$fitted.values)
  R.l <-  1 -  dev / nullDev
  R.cs <- 1- exp ( -(nullDev - dev) / modelN)
  R.n <- R.cs / ( 1 - ( exp (-(nullDev / modelN))))
  R.hlp <- (ResourceSelection::hoslem.test(LogModel$y, fitted(LogModel), g=10))$p.value
  cat("Pseudo R^2 for logistic regression\n")
  cat("McFadden's R^2             ", round(R.l, 3), "\n")
  cat("Cox and Snell R^2          ", round(R.cs, 3), "\n")
  cat("Nagelkerke R^2             ", round(R.n, 3),    "\n")
  cat("Hosmer and Lemeshow p-value", round(R.hlp, 3),    "\n")
} 

logisticPseudoR2s(model2)
logisticPseudoR2s(model15)


# Comparing all the models using a loop

models <- list(model0, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15)

R2table <- data.frame()
for (i in 1:16) {
  R2table[i,1] <- paste('Model', i-1, sep="", collapse=NULL)
  dev <- models[[i]]$deviance
  nullDev <- models[[i]]$null.deviance
  modelN <- length(models[[i]]$fitted.values)
  R.l <-  1 -  dev / nullDev
  R.cs <- 1- exp ( -(nullDev - dev) / modelN)
  R.n <- R.cs / ( 1 - ( exp (-(nullDev / modelN))))
  R2table[i,2] <-  round(1 -  dev / nullDev, 3) # McFadden's Pseudo-R2
  R2table[i,3] <- round(1- exp ( -(nullDev - dev) / modelN), 3) # Cox and Snell R^2
  R2table[i,4] <- round(R.cs / ( 1 - ( exp (-(nullDev / modelN)))) ,3)# Nagelkerke R^2 
  R2table[i,5] <- round((ResourceSelection::hoslem.test(models[[i]]$y, fitted(models[[i]]), g=10))$p.value, 3) # Hosmer and Lemeshow p-value
}

colnames(R2table) <- c("Model_ID", "McFaddens R^2", "Cox and Snell R^2", "Nagelkerke R^2", "Hosmer Lemeshow p-value")
R2table

# We choose to proceed with model15, because it ranks highly in all Goodness-of-Fit tests and it has the highest Hosmer & Lemeshow p-value, meaning there is not evidence the model has a poor fit.

summary(model15)

# Confusion Matrix (Accuracy)
prediction <- ifelse(predict(model15, type="response")>0.5,1,0)
confusion_matrix<-table(dataset$Churn, prediction)
round(prop.table(confusion_matrix), 2)


### MODEL ASSUMPTIONS

summary(model15)

# Multicolinearity
# We have covered this, using the VIFs

# Linearity of the logit
# We create interaction terms of the variables with their logs.

data<-data.frame(Churn=dataset$Churn, CustServ.Calls=dataset$CustServ.Calls, Intl.Calls=dataset$Intl.Calls, Intl.Charge=dataset$Intl.Charge, Domestic.Charge=data13$Domestic.Charge, Intl.l.Plan=dataset$Int.l.Plan, VMail.Plan=dataset$VMail.Plan)

data$logCustServ.Calls <- log(1+data$CustServ.Calls)*data$CustServ.Calls
data$logIntl.Calls <- log(1+data$Intl.Calls)*data$Intl.Calls
data$logIntl.Charge <- log(1+data$Intl.Calls)*data$Intl.Charge
data$logDomestic.Charge <- log(data$Intl.Calls)*data$Domestic.Charge

InteractionModel <- glm(Churn~., data = data, family = "binomial")
summary(InteractionModel)

# Independence of Error Terms

par(mfrow = c(2,4), mar = c(5,5,1,1))
plot(dataset$CustServ.Calls, resid(model15, type = 'pearson'), ylab = 'Residuals (Pearson)', xlab = 'CustServ.Calls', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'blue', cex = 1.5)
plot(dataset$Intl.Calls, resid(model15, type = 'pearson'), ylab = 'Residuals (Pearson)', xlab = 'Intl.Calls', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'blue', cex = 1.5)
plot(dataset$Intl.Charge, resid(model15, type = 'pearson'), ylab = 'Residuals (Pearson)', xlab = 'Intl.Charge', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'blue', cex = 1.5)
plot(data13$Domestic.Charge, resid(model15, type = 'pearson'), ylab = 'Residuals (Pearson)', xlab = 'Domestic.Charge', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'blue', cex = 1.5)

plot(dataset$CustServ.Calls, resid(model15, type = 'deviance'), ylab = 'Residuals (Deviance)', xlab = 'CustServ.Calls', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'red', cex = 1.5)
plot(dataset$Intl.Calls, resid(model15, type = 'deviance'), ylab = 'Residuals (Deviance)', xlab = 'Intl.Calls', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'red', cex = 1.5)
plot(dataset$Intl.Charge, resid(model15, type = 'deviance'), ylab = 'Residuals (Deviance)', xlab = 'Intl.Charge', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'red', cex = 1.5)
plot(data13$Domestic.Charge, resid(model15, type = 'deviance'), ylab = 'Residuals (Deviance)', xlab = 'Domestic.Charge', cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'red', cex = 1.5)

dev.off()


### MODEL INTERPRETATION ###

confint(model15) # Asymptotic 95% C.I. for model coefficients
wald.test(b = coef(model15), Sigma = vcov(model15), Terms = 4:6)

exp(coef(model15)) # Odds ratio scale
exp(cbind(OR = coef(model15), confint(model15)))

# Centered covariates

dataset2<-data.frame(CustServ.Calls=dataset$CustServ.Calls, Intl.Calls=dataset$Intl.Calls, Intl.Charge=dataset$Intl.Charge, Domestic.Charge=data13$Domestic.Charge)
dataset2 <- as.data.frame(scale(dataset2, center = TRUE, scale = F))
dataset2$Churn<-dataset$Churn
dataset2$Intl.l.Plan <- dataset$Int.l.Plan
dataset2$VMail.Plan <- dataset$VMail.Plan
str(dataset2)

attach(dataset2)
cen.model<-glm(Churn~., data = dataset2, family = "binomial")
summary(cen.model)
