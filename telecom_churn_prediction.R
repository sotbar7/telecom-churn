#########################################
#   TELECOM CHURN PREDICTION PROJECT    #
#   SOTIRIS BARATSAS                    #
#   Contact: sotbaratsas[at]gmail.com   #
#########################################

library(glmnet)
library(aod)
library(psych)
library(car)
library(corrgram)
library(nnet)
library(class)
library(tree)
library(MASS)
library(penalizedLDA)
library(cluster)


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

# Using Cook's Distance to identify leverage points and possibly remove some observations to improve the model
cooksd <- cooks.distance(model0)
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
numcolumns <- numcolumns[-influential, ]
fcolumns <- fcolumns[-influential, ]
length(dataset[,1])



### VARIABLE TRANSFORMATIONS ###
# Because a log or polynomial transformation would have negative consequences on our ability to interpret the model in a practical way, we will only perform some simple (aggregate) transformations

# Feature Engineering
dataset$Tot.Mins <- dataset$Day.Mins + dataset$Eve.Mins + dataset$Night.Mins
dataset$Tot.Calls <- dataset$Day.Calls + dataset$Eve.Calls + dataset$Night.Calls
dataset$Tot.Charge <- dataset$Day.Charge + dataset$Eve.Charge + dataset$Night.Charge



##################
# CLASSIFICATION #
##################

### Turning Account Length into a factor

# summary(dataset$Account.Length)
# findInterval(dataset$Account.Length, c(0, 50, 100, 150, 200))
# cut(dataset$Account.Length, breaks=5, right = FALSE)
# dataset$Account.Length <- cut(dataset$Account.Length, breaks=c(0, 50, 100, 150, 200, 243), right = FALSE)
# head(dataset$Account.Length)
# colnames(dataset) <- make.names(colnames(dataset))
# dataset <- na.omit(dataset)

### Scaling numeric variables

index <- sapply(dataset, class) != "factor"
dataset[index] <- lapply(dataset[index], scale)
str(dataset)

data<-dataset
# ### Getting dummy variables for different factor levels
data <- fastDummies::dummy_cols(dataset)
dropcols <- c("Intl.l.Plan", "VMail.Plan", "State", "Area.Code", "Gender", "Churn_0", "Churn_1")
# dropcols <- c("Account.Length", "Intl.l.Plan", "VMail.Plan", "State", "Area.Code", "Gender", "Churn_0", "Churn_1")
data <- data[, ! names(data) %in% dropcols, drop = F]
str(data)


### Train / Test Split ###

set.seed(8294)
library(caret)
tr_index <- sample(nrow(data), 0.7*nrow(data), replace = FALSE)

train <- data[tr_index,]
test <- data[-tr_index,]

# Saving the values of the 'Churn' column for the test dataset and then dropping it
y_test <- test$Churn
test$Churn <- NULL

# Checking the dimensions of the train & test datasets
dim(train)
dim(test)
prop.table(table(y_test))

attach(train)
View(train)

### LOGISTIC REGRESSION CLASSIFIER ###

LogModel <- glm(Churn ~ ., family=binomial(link="logit"), data=train)
print(summary(LogModel))
anova(LogModel, test="Chisq")

LogModel_pred <- predict(LogModel,newdata=test,type='response')
LogModel_pred <- round(LogModel_pred, 0)
misClasificError <- mean(LogModel_pred != y_test)
print(paste('Logistic Regression Accuracy',1-misClasificError))
confusionMatrix(factor(LogModel_pred), y_test, positive = "1")

### DECISION-TREE 

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(8294)
library(e1071)

dtree_fit <- train(x=train[,-1], y = train$Churn, method = "rpart",
                   parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)

dtree_fit

library(rpart.plot)
prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)

predict(dtree_fit, newdata = test[1,])

tree_pred <- predict(dtree_fit, newdata = test)
confusionMatrix(tree_pred, y_test, positive = "1")


dtree_fit_gini <- train(x=train[,-1], y = train$Churn, method = "rpart",
                          parms = list(split = "gini"),
                          trControl=trctrl,
                          tuneLength = 10)
dtree_fit_gini

prp(dtree_fit_gini$finalModel, box.palette = "Reds", tweak = 1.2)

gini_pred <- predict(dtree_fit_gini, newdata = test)
confusionMatrix(gini_pred, y_test, positive = "1")

### RANDOM FOREST MODEL
library(randomForest)
# Create a Random Forest model with default parameters
rfmodel <- randomForest(x=train[,-1], y = train$Churn, importance = TRUE)
rfmodel

rfmodel2 <- randomForest(Churn ~ ., data = train, ntree = 500, mtry = 7, importance = TRUE)
rfmodel2

# Predicting on train set
predTrain <- predict(rfmodel, train, type = "class")
# Checking classification accuracy
table(predTrain, train$Churn)  

# Predicting on Validation set
rf_pred <- predict(rfmodel2, test, type = "class")
# Checking classification accuracy
mean(rf_pred == y_test)                    
confusionMatrix(rf_pred, y_test,positive = "1")

# To check important variables
importance(rfmodel2)        
varImpPlot(rfmodel2) 


# Using For loop to identify the right mtry for model
a=c()
i=5
for (i in 3:9) {
  rfmodel3 <- randomForest(Churn ~ ., data = train, ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(rfmodel3, test, type = "class")
  a[i-2] = mean(predValid == y_test)
}
a
plot(3:9, a)

rfmodel3 <- randomForest(Churn ~ ., data = train, ntree = 500, mtry = 9, importance = TRUE)
# Predicting on Validation set
rf_pred <- predict(rfmodel3, test, type = "class")
# Checking classification accuracy
mean(rf_pred == y_test)                    
confusionMatrix(rf_pred, y_test, positive = "1")


## COMPARING BETWEEN THE 2 MODELS

# We will compare model 1 of Random Forest with Decision Tree model

model_dt = train(Churn ~ ., data = train, method = "rpart")
model_dt_1 = predict(model_dt, data = train)
table(model_dt_1, train$Churn)

mean(model_dt_1 == train$Churn)

model_dt_vs = predict(model_dt, newdata = test)
table(model_dt_vs, y_test)
mean(model_dt_vs == y_test)


### NAIVE BAYES ###

library('e1071')
nbm <- naiveBayes(y = train$Churn, x = train[,-1])
nbclass <- predict(nbm, newdata=test)
mean(nbclass == y_test)                    
confusionMatrix(nbclass, y_test, positive = "1")


### K-nearest neighbors ###
library('class')
km1<-knn(train = train[,-1], test = test, cl = train[,1], k = 4)
mean(km1 == y_test)                    
confusionMatrix(km1, y_test, positive = "1")

km2<-knn(train = train[,-1], test = test, cl = train[,1], k = 6)
mean(km2 == y_test)                    
confusionMatrix(km2, y_test, positive = "1")

### SVM ###

svm_model <- svm(Churn ~ ., data = train)
svm_pred <- predict(svm_model, test)
mean(svm_pred == y_test)                    
confusionMatrix(svm_pred, y_test, positive = "1")

### XGBOOST ###
library(xgboost)
# Converting our data frame into a sparse matrix
train_matrix <- data.matrix(train)
train_response <- as.numeric(as.character(train$Churn))
classifier = xgboost(data = train_matrix[,-1], label = train_response, max.depth = 5, eta=0.5, nrounds = 10, objective = "binary:logistic")

# Predicting the Test set results
xgb_pred <- predict(classifier, newdata = data.matrix(test))
xgb_pred <- round(xgb_pred, 0)
mean(xgb_pred == test_response)                    
confusionMatrix(factor(xgb_pred), y_test, positive = "1")

# AdaBoost

library(fastAdaboost)
ada_model = adaboost(Churn~., data=train, 10)
ada_pred = predict(ada_model, newdata=test)$class
mean(ada_pred == y_test)                    
confusionMatrix(ada_pred, y_test, positive = "1")


####################
# MODEL EVALUATION #
####################

# ARI: Adjusted Rand Index
# Comparing performance according to Adjusted Rand Index with the true class

allmodels<- cbind(LogModel_pred, tree_pred, gini_pred, rf_pred, nbclass, km2, svm_pred, xgb_pred, ada_pred)
colnames(allmodels) <- c('M-Logistic', 'Decision-Tree', 'Decision-Tree (gini)', 'Random Forest', 'Naive Bayes', 
                         'K-means 6', 'SVM', 'XGBoost', 'Adaboost')
#	computing adjusted Rand Index for all models
library('mclust')
ari <- apply(allmodels,2,function(x){adjustedRandIndex(x, y_test)} )

par(mar = c(4,6,3,1))
barplot(ari[order(ari, decreasing = F)], 
        horiz=TRUE, las = 1, xlab = 'Adjusted Rand Index', xlim = c(0,1), cex.names=0.65, xpd=F)


# ROC & AUC
library(pROC)
auc_list <- apply(allmodels,2,function(x){auc(roc(y_test, x))} )
par(mar = c(4,6,3,1))
barplot(auc_list[order(auc_list, decreasing = F)], 
        horiz=TRUE, las = 2, xlab = 'Area Under the Curve (AUC)', xlim = c(0,1), cex.names=0.65)


##############
# CLUSTERING #
##############

### Filtering only the variables that relate to usage (exlude demographics, charges and account length)
keepcols <- c("Churn", "VMail.Message", "Day.Mins", "Eve.Mins", "Night.Mins", "Intl.Mins", "CustServ.Calls", "VMail.Plan", "Int.l.Plan", "Day.Calls", "Eve.Calls", "Night.Calls", "Intl.Calls")
train <- dataset[, names(dataset) %in% keepcols, drop = F]
str(train)
attach(train)


### HIERARCHICAL CLUSTERING ###

library(corrgram)
corrgram(train[,-1])
pairs(train[,-1], col=train[,1])

# Clustering using Ward's distance and all the numeric variables
hc1 <- hclust(dist(train[,-c(1,8,9)]), method="ward.D") 
summary(hc1)

clas<-cutree(hc1,3)
par(mfrow=c(1,1))
clusplot(train, clas, color=TRUE, shade=TRUE, labels=2, lines=0)

par(mfrow=c(1,1))
plot(hc1)
rect.hclust(hc1,3) ## puts a rectiangular arounf the groups
plot(silhouette(clas,dist(train[,-c(1,8,9)])))

# Other Linkage Types
hc2<-hclust(dist(train[,-c(1,8,9)]),method="complete")
hc3<-hclust(dist(train[,-c(1,8,9)]),method="single")
hc4<-hclust(dist(train[,-c(1,8,9)]),method="average")
hc5<-hclust(dist(train[,-c(1,8,9)]),method="cen")

# Silhouette values
clas1<-cutree(hc1,3)
clas2<-cutree(hc1,3)
clas5<-cutree(hc1,3)

table(clas1, train$Churn)

par(mfrow=c(1,1))
plot(silhouette(clas1, dist(train[,-c(1,8,9)])))
sil<-silhouette(clas1, dist(train[,-c(1,8,9)]))
mean(sil[,3])


# Average Silhouette Values

clas1<-cutree(hc1, k=2:9)

res<-NULL
for (i in 1:8){
  a<-silhouette(clas1[,i], dist(train[,-c(1,8,9)]))
  res<-c(res,mean(a[,3]))
}
plot(2:9,res,type="b",ylim=c(0,0.5),xlab="clusters", main="Average Silhouette")

# Determining the optimal number of variables using ARI (Adj Rand Index)
indices<-NULL
for  ( i in 3:12) {
  usevar<- train[,2:i]
  myclust<-hclust(dist(usevar),method="ward.D")
  myclass<- cutree(myclust,3)
  indices<-c(indices,adjustedRandIndex(myclass,train[,1]))
}
plot(indices)

# USING MAHALANOBIS DISTANCE

mydist<- as.dist(apply(train[,-c(1,8,9)], 1, function(i) mahalanobis(train[,-c(1,8,9)], i, cov = cov(train[,-c(1,8,9)]))))
hclab<- hclust(mydist, method="ward.D")
plot(hclab)
rect.hclust(hclab,3)

# Determining the optimal number of clusters
mahcl <- cutree(hclab, k=2:9)
res<-NULL
for (i in 1:8){
  a<-silhouette(mahcl[,i], mydist)
  res<-c(res,mean(a[,3]))
}
plot(2:9,res,type="b",ylim=c(0,0.5),xlab="clusters", main="Average Silhouette")

mahcl <- cutree(hclab, 3)
par(mfrow=c(1,1))
clusplot(train, mahcl, color=TRUE, shade=TRUE, labels=2, lines=0)


# USING GOWER DISTANCE

dist2<-daisy(data.matrix(train[,-1]))
hcgower <- hclust(dist2, method="ward.D")

gowercl <- cutree(hcgower, k=2:9)
res<-NULL
for (i in 1:8){
  a<-silhouette(gowercl[,i], mydist)
  res<-c(res,mean(a[,3]))
}
plot(2:9,res,type="b",ylim=c(0,0.5),xlab="clusters", main="Average Silhouette")

clas<-cutree(hcgower,2)
par(mfrow=c(1,1))
plot(hcgower)
rect.hclust(hcgower,2) ## puts a rectiangular arounf the groups
clusplot(train, clas, color=TRUE, shade=TRUE, labels=2, lines=0)
plot(silhouette(clas,dist(train[,-c(1)])))

# USING FEWER VARIABLES

mydist<- as.dist(apply(train[,c(2,7)], 1, function(i) mahalanobis(train[,c(2,7)], i, cov = cov(train[,c(2,7)]))))
hclab<- hclust(mydist, method="ward.D")
rect.hclust(hclab,3)
clasb<-cutree(hclab,3)
plot(train[,c(2,7)],col=clasb)

### 

dist3<-daisy(data.matrix(train[,c(2,7)]))
hcgower <- hclust(dist3, method="ward.D")

clas<-cutree(hcgower,3)
par(mfrow=c(1,1))
plot(hcgower)
rect.hclust(hcgower,3) ## puts a rectiangular arounf the groups


### K-MEANS ###
# We want small within sum of squares and big between sum of squares

kmeans(train, 3, trace=3, iter.max=10)

# RUNNING WITH 2-15 CLUSTERS TO CHOOSE THE BEST

within<-NULL

for (i in 2:15) {
  within<-c(within,kmeans(train[,-1],i,nstart=20)$tot.withinss) }
par(mfrow=c(1,1))
plot(2:15,within, type="b",xlab="number of cluster", ylab="total within ss")


par(mfrow=c(1,1))
i <- 2
mod<-kmeans(train[,-1],i,nstart=20)
plot(train[,-1],col=mod$cluster, main=paste(i," clusters"))
print(paste("Within Sum of Squares = ", mod$tot.withinss))
print(paste("Between Sum of Squares = ", mod$betweenss))
i <- 3
mod<-kmeans(train[,-1],i,nstart=20)
plot(train,col=mod$cluster, main=paste(i," clusters"))
print(paste("Within Sum of Squares = ", mod$tot.withinss))
print(paste("Between Sum of Squares = ", mod$betweenss))
i <- 4
mod<-kmeans(train[,-1],i,nstart=20)
plot(train,col=mod$cluster, main=paste(i," clusters"))
print(paste("Within Sum of Squares = ", mod$tot.withinss))
print(paste("Between Sum of Squares = ", mod$betweenss))
i <- 5
mod<-kmeans(train[,-1],i,nstart=20)
plot(train,col=mod$cluster, main=paste(i," clusters"))
print(paste("Within Sum of Squares = ", mod$tot.withinss))
print(paste("Between Sum of Squares = ", mod$betweenss))


clusplot(train, mod$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

### CLUSTERING WITH PRINCIPAL COMPONENTS

### PRINCIPAL COMPONENT ANALYSIS ###

eigen(cov(numcolumns))   # eigen values and eigenvectors from covariance matrix
eigen(cor(numcolumns))

data.pca2 <- princomp(numcolumns,cor=TRUE,scores=TRUE)
summary(data.pca2)
str(data.pca2)
head(data.pca2$scores)

a1<-solve(data.pca2$loadings)
nX<-data.pca2$scores%*%a1  #fully reconstruct the data
plot(numcolumns[,1],nX[,1])   

apply(nX,2,mean)
apply(nX,2,var)
nnX<-scale(numcolumns,apply(numcolumns,2,mean),apply(numcolumns,2,sd))
apply(nnX,2,mean)
apply(nnX,2,var)

plot(nnX[,1],nX[,1]) 

# Keeping less PC
# keep only 2
nX<-data.pca2$scores[,-c(3:15)]%*%a1[-c(3:15),]
plot(numcolumns[,1],nX[,1])

# keep only 3
nX<-data.pca2$scores[,-c(4:15)]%*%a1[-c(4:15),]
plot(numcolumns[,1],nX[,1])

# keep only 7
nX<-data.pca2$scores[,-c(8:15)]%*%a1[-c(8:15),]
plot(numcolumns[,1],nX[,1])

# Clustering

# Using the Within Sum of Squares to find the optimal number of clusters
within<-NULL
for (i in 2:15) {
  within<-c(within,kmeans(nX[,-1],i,nstart=20)$tot.withinss) }
par(mfrow=c(1,1))
plot(2:15,within, type="b",xlab="number of cluster", ylab="total within ss")

i <- 4
mod<-kmeans(nX[,-1],i,nstart=20)
plot(data.frame(nX),col=mod$cluster, main=paste(i," clusters"))
print(paste("Within Sum of Squares = ", mod$tot.withinss))
print(paste("Between Sum of Squares = ", mod$betweenss))

