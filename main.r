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




set.seed(2)

# load data
data(diamonds)
dim(diamonds)  ## 53940

# separate numerical and non numerical columns
num_cols <- select_if(diamonds, is.numeric)   
cat_col <- select_if(diamonds, negate(is.numeric))

#check outliers
par(mfrow=c(2,4))
for (i in colnames(num_cols)) {
  boxplot(diamonds[,i], main = i)
}



# remove outliers ## trying to put into for loop, but this works for now
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

cor_mat <- cor(diamonds[, -c(2:4)])
cor_mat
corrplot(cor_mat, method="pie", type="lower", addCoef.col = "black")

# check outliers again, should not be present now
par(mfrow=c(2,4))
for (i in colnames(num_cols)) {
  boxplot(diamonds[,i], main = i)
}

dim(diamonds) #46532 removing outliers removed % data

# encode the non-numeric values
diamonds$cut = as.numeric(unclass(diamonds$cut))
diamonds$clarity = as.numeric(unclass(diamonds$clarity))
diamonds$color = as.numeric(unclass(diamonds$color))



# reduce dimensionality -  pca and correlation

# correlation plot
ggpairs(diamonds)

# pca
pr.out <- prcomp(diamonds %>% select (-price), scale = TRUE)

std_dev <- pr.out$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)

plot(prop_varex, xlab = "Principal Components",
     ylab = "Proportion of Variance", type = "b")
plot(cumsum(prop_varex), xlab = "Principal Components",
     ylab = "Cumulative Proportion of Variance",
     type = "b")

# we remove x, y and z variables based on pca and correlation plot output
diamonds <- diamonds %>% 
  select (-x, -y, -z)

# split into 70% train and 30% test
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


# apply regression methods

# linear regression
#fit.lm <-lm(train_Y ~ ., data = train_X)

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

########################################

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

# subset selection on train data
fit.subset.train <- regsubsets(train_data$price ~ ., data = train_data, method = "exhaustive")
summary(fit.subset.train)

model.ids <- 1:6
cv.train.errors <-  map(model.ids, formula, fit.subset.train, "price") %>%
  map(error, train_data) %>%
  unlist()

#min train mse for model will all 6 predictors
min(cv.train.errors)  #772.2178

#coef of train model with min RMSE
coef(fit.subset.train, which.min(cv.train.errors))

#(Intercept)       carat         cut       color     clarity       depth       table 
#2999.47335  2740.00485    47.62913  -367.05082   604.05949   -14.98773   -37.16489 

# subset selection on test data
fit.subset.test <- regsubsets(test_data$price ~ ., data = test_data, method = "exhaustive")
summary(fit.subset.test)

model.ids <- 1:6
cv.test.errors <-  map(model.ids, formula, fit.subset.test, "price") %>%
  map(error, test_data) %>%
  unlist()

#min test mse for model will all 6 predictors 
min(cv.test.errors) #770.3904


########################################

## Ridge Regression

cv.ridge <- cv.glmnet(as.matrix(train_X), train_Y, alpha = 0)
best.lambda.ridge <- cv.ridge$lambda.1se
best.lambda.ridge #240.2071
ridge.model <- glmnet(as.matrix(train_X), train_Y,
                      alpha = 0, lambda = best.lambda.ridge)
coef(ridge.model)

#               s0
#(Intercept) 2999.473351
#carat       2417.869601
#cut           43.392618
#color       -250.570431
#clarity      442.393856
#depth         -9.379870
#table         -3.515143


# prediction on train data
ridge.train.pred <- predict(ridge.model, as.matrix(train_X))
ridge.train.MSE <- mean((ridge.train.pred - train_Y)^2)
ridge.train.MSE #677572.2

#prediction on test data
ridge.test.pred <- predict(ridge.model, as.matrix(test_X))
ridge.test.MSE <- mean((ridge.test.pred - test_Y)^2)
ridge.test.MSE #668929.1


########################################

## Lasso Regression

cv.lasso <- cv.glmnet(as.matrix(train_X), train_Y, alpha = 1)
best.lambda.lasso <- cv.lasso$lambda.1se
best.lambda.lasso #33.26595
lasso.model <- glmnet(as.matrix(train_X), train_Y,
                      alpha = 1, lambda = best.lambda.lasso)
coef(lasso.model)

#             s0
#(Intercept) 2999.473351
#carat       2661.196023
#cut           32.638243
#color       -311.373507
#clarity      547.348751
#depth          .       
#table         -1.411057

# prediction on train data
lasso.train.pred <- predict(lasso.model, as.matrix(train_X))
lasso.train.MSE <- mean((lasso.train.pred - train_Y)^2)
lasso.train.MSE #604506.6

#prediction on test data
lasso.test.pred <- predict(lasso.model, as.matrix(test_X))
lasso.test.MSE <- mean((lasso.test.pred - test_Y)^2)
lasso.test.MSE #599609.9

########################################

## Regression Tree

fit.tree <- rpart(train_Y ~ ., data = train_X)
fit.tree$cptable

#CP nsplit rel error    xerror        xstd
#1 0.71027869      0 1.0000000 1.0000325 0.008671541
#2 0.06266607      1 0.2897213 0.2899212 0.002929652
#3 0.04993119      2 0.2270552 0.2277476 0.002427827
#4 0.03393171      3 0.1771241 0.1780637 0.002382428
#5 0.01511627      4 0.1431923 0.1432967 0.001808565
#6 0.01492336      5 0.1280761 0.1324440 0.001696824
#7 0.01023770      6 0.1131527 0.1132481 0.001480342
#8 0.01000000      7 0.1029150 0.1038572 0.001461335

best.cp <- fit.tree$cptable %>%
  as_tibble() %>%
  filter(xerror == min(xerror)) %>%
  head(1) %>%
  pull(CP)
best.cp #0.01 because it has min xerror

#prune tree with best cp
prune.tree <- prune(fit.tree, cp = best.cp)
rpart.plot(prune.tree)

# prediction on train data
prune.train.pred <- predict(
  prune.tree,
  train_X
)

#train error
mean((prune.train.pred - train_Y)^2) #694893.6

# prediction on test data
prune.test.pred <- predict(
  prune.tree,
  test_X
)

#test error
mean((prune.test.pred - test_Y)^2) #692470.4

########################################


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

