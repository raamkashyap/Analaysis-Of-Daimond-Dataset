library(tidyverse)
library(rpart)
library(dplyr)
library(rpart.plot)
library(ISLR2)
library(plyr)
library(ggplot2)
library(GGally)
library(leaps)
library(randomForest)
library(Metrics)
library(caret)
library(leaps)
library(glmnet)
library(glmnetUtils)
library(pls) 
library(boot)
library(corrplot)

# set seed for reproducible results.
set.seed(2)


# Load Dataset
data(diamonds)

# view dataset summary
summary(diamonds)

# view a few rows of the data
head(diamonds)

# Numerical Columns
num_cols <- select_if(diamonds, is.numeric)
colnames(num_cols)

# Categorical Column
cat_col <- select_if(diamonds, negate(is.numeric))
colnames(cat_col)

# Outlier Detection and Filtering

# boxplot before outlier removal
par(mfrow=c(2,4))
for (i in colnames(num_cols)) {
  boxplot(diamonds[,i], main = i)
}

# outlier filtering
outliers.carat <- boxplot(diamonds$carat, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$carat %in% outliers.carat),]
outliers.depth <- boxplot(diamonds$depth, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$depth %in% outliers.depth),]
outliers.table <- boxplot(diamonds$table, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$table %in% outliers.table),]
outliers.price <- boxplot(diamonds$price, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$price %in% outliers.price),]
outliers.x <- boxplot(diamonds$x, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$x %in% outliers.x),]
outliers.y <- boxplot(diamonds$y, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$y %in% outliers.y),]
outliers.z <- boxplot(diamonds$z, plot=FALSE)$out
diamonds <- diamonds[-which(diamonds$z %in% outliers.z),]

# boxplot after outlier removal
par(mfrow=c(2,4))
for (i in colnames(num_cols)) {
  boxplot(diamonds[,i], main = i)
}

# After outlier removal, we have retained approximately 47000 records.

# Data Encoding - Convert categorical variables to numerical encoding.
diamonds$cut = as.numeric(unclass(diamonds$cut))
diamonds$clarity = as.numeric(unclass(diamonds$clarity))
diamonds$color = as.numeric(unclass(diamonds$color))

# Pairwise association plot
ggpairs(diamonds, progress = FALSE)

# Plotting correlations as pie charts
cor_mat <- cor(diamonds[, -c(2:4)])
corrplot(cor_mat, method="pie", type="lower", addCoef.col = "black")

# Principal Component Analysis
pr.out <- prcomp(diamonds %>% select (-price), scale = TRUE)
std_dev <- pr.out$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)

# scree plot
plot(prop_varex, xlab = "Principal Components",
     ylab = "Proportion of Variance", type = "b")

# remove x, y, z from dataset as per pca and correlation plot
diamonds <- diamonds %>% 
  select (-x, -y, -z)

# Data Split
partition = sample(nrow(diamonds), as.integer(nrow(diamonds)*0.7))
train_data <- diamonds[partition,]
test_data <- diamonds[-partition,]

# normalize/standardize the data
mean_vector = apply(train_data %>% select (-price), MARGIN=2, mean)
sd_vector = apply(train_data %>% select (-price), MARGIN=2, sd)
train_data[c(-7)] <- scale(train_data[c(-7)], center=mean_vector, scale=sd_vector)
test_data[c(-7)] <- scale(test_data[c(-7)], center=mean_vector, scale=sd_vector)

# Separate out predictors and response
train_X <- train_data %>%
  select(-price)
train_Y <- train_data$price
test_X <- test_data %>%
  select(-price)
test_Y <- test_data$price


# Prediction Algorithms

# Linear Regression

# Applying linear regression model using price as the target and other features as the feature space
lin_reg_obj <- lm(train_Y ~ ., data = train_X)
# Compute the predictions
predict_price_lin_reg <- predict(lin_reg_obj, test_X)
#Print MAE
print(mae(test_Y,predict_price_lin_reg))  # 561.7
# Print MSE
postResample(predict_price_lin_reg,test_Y)['RMSE']^2  #593280.4 
# Calculating the R-squared value
postResample(predict_price_lin_reg,test_Y)['Rsquared'] # 0.912


########################################################################


# Ridge Regression

# Find best lambda using cross validation
cv.ridge <- cv.glmnet(as.matrix(train_X), train_Y, alpha = 0)
best.lambda.ridge <- cv.ridge$lambda.1se
print(paste("Best lambda: ", best.lambda.ridge))

# Fit a model using best lambda value
ridge.model <- glmnet(as.matrix(train_X), train_Y,
                      alpha = 0, lambda = best.lambda.ridge)

# Predictor importance
coef(ridge.model)


# Prediction on train data
ridge.train.pred <- predict(ridge.model, as.matrix(train_X))
print(paste("MSE on train data: ", mean((ridge.train.pred - train_Y)^2)))

# Prediction on test data
ridge.test.pred <- predict(ridge.model, as.matrix(test_X))
print(paste("MSE on test data: ", mean((ridge.test.pred - test_Y)^2)))


########################################################################


# LASSO Regression

# Find best lambda using cross validation
cv.lasso <- cv.glmnet(as.matrix(train_X), train_Y, alpha = 1)
best.lambda.lasso <- cv.lasso$lambda.1se
print(paste("Best lambda: ", best.lambda.lasso)) #33.26595

# Fit a model using best lambda value
lasso.model <- glmnet(as.matrix(train_X), train_Y,
                      alpha = 1, lambda = best.lambda.lasso)

# Predictor importance
coef(lasso.model)

# prediction on train data
lasso.train.pred <- predict(lasso.model, as.matrix(train_X))
print(paste("MSE on train data: ", mean((lasso.train.pred - train_Y)^2)))

#prediction on test data
lasso.test.pred <- predict(lasso.model, as.matrix(test_X))
print(paste("MSE on test data: ", mean((lasso.test.pred - test_Y)^2)))


########################################################################


# Regression Tree


fit.tree <- rpart(train_Y ~ ., data = train_X)

# cp table for model
fit.tree$cptable

# Best cp with minimum xerror
best.cp <- fit.tree$cptable %>%
  as_tibble() %>%
  filter(xerror == min(xerror)) %>%
  head(1) %>%
  pull(CP)
print(paste("Best cp for prune tree ", best.cp))

# Prune tree with best cp
prune.tree <- prune(fit.tree, cp = best.cp)
rpart.plot(prune.tree)

# prediction on train data
prune.train.pred <- predict(
  prune.tree,
  train_X
)

#train error
print(paste("MSE on train data: ", mean((prune.train.pred - train_Y)^2)))

# prediction on test data
prune.test.pred <- predict(
  prune.tree,
  test_X
)

#test error
print(paste("MSE on test data: ", mean((prune.test.pred - test_Y)^2)))


########################################################################


# Bootstrap

# Applying bootstrapping using multiple linear regression
# Creating a function to caluculate the coeffecients that have been fitted
bootstrap_func <- function(formula, data_frame, obs)
{
  bootstrap_sample <-  data_frame[obs, ] # Select a sample using bootstrap
  lin_reg_model <- lm(formula, data = bootstrap_sample) # Applying linear regression on the bootstrap sample 
  return(coef(lin_reg_model)) # The function should return the weights
}
# Apply bootstrap for 1001 samples
bootstrap_final <- boot(data = diamonds, statistic=bootstrap_func, R = 1001, formula = price ~ .)
# Plot the distribution of the bootstrapped samples
plot(bootstrap_final,index = 1) # Intercept of the model
plot(bootstrap_final,index = 2) # Carat predictor variable
plot(bootstrap_final,index = 3) # Cut predictor variable
plot(bootstrap_final,index = 4) # Colour predictor variable
plot(bootstrap_final,index = 4) # Clarity predictor variable
plot(bootstrap_final,index = 5) # Depth predictor variable
plot(bootstrap_final,index = 6) # Table predictor variable


########################################################################


# Random Forest Regression

#Applying Random Forest algorithm

#install.packages("randomForest")
random_forest_model <- randomForest(x = train_X, y = train_Y, ntree = 100)
# Predicting on test set
predict_rf <- predict(random_forest_model, test_X)
#Displaying the testing error
#install.packages('Metrics')
# Calculating the mean absolute error
print(mae(test_Y,predict_rf))  # 222.24
# Calculating the  mean square error
postResample(predict_rf,test_Y)['RMSE']^2 # 140845.7
# Calculating the R^2 value
postResample(predict_rf,test_Y)['Rsquared'] # 0.97
plot(random_forest_model)
