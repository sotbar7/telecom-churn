# Telecom-Churn

PART 1: Understanding Customer Churn in Telecom (Descriptive)

In the first part of this study, we attempted to formulate a regression model that identifies the characteristics that influence whether a customer is probable to switch telecommunications providers (Churn). We started with a logistic regression model that made use of all variables, performed variable selection using AIC and BIC through the stepwise method and moved on to other models, using LASSO, or a few simple (aggregate) transformation of certain predictor variables. We concluded that the best logistic regression model we could find was produced with an aggregate transformation of the variables that concern domestic charges for various times of the day (Day Charges, Evening Charges, Night Charges).

PART 2: Predicting Customer Churn in Telecom (Predictive)

In the second part of the study, we attempted to formulate a predictive model that identifies whether a customer is probable to switch telecommunications providers (Churn) or stay with the company. We started with a Logistic Regression classifier, and moved on to methods such as Decision Tree, Random Forest, XGBoost, Adaboost, SVM, KNN and Naive- Bayes. We concluded that the best predictive model we could find was XGBoost, which manages to identify correctly almost all the non-churners and the vast majority of the churners. Closely trailing was the Decision Tree model, which is more easily interpretable and applicable in real business problems.

On the other hand, Cluster Analysis was a bit more challenging. The Hierarchical Clustering methods we used weren’t very effective. Using the Mahalanobis distance and the Gower distance, we managed to produce 2 clustering methods with Silhouette values equal to 0.2. Using the K-Means method, the results became a little bit better, especially using Principal Components and creating 4 clusters.
