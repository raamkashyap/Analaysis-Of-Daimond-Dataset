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

plot(prop_varex, type = "b")

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

# Print RMSE
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

bootstrap_final <- boot(data = diamonds, statistic=bootstrap_func, R = 1001, formula = price ~ .b)

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

### principle component regression

fit.pcr <- pcr(train_Y ~ .
               , data = train_X, scale = FALSE, center = FALSE)


# prediction on train data
pred.train.list <- predict(fit.pcr)
best.train.M = 0.
best.train.MSE = Inf
best.train.coefficients = NULL
for (M in 1:ncol(train_X)) {
  pred = pred.train.list[, , M]
  mse = mean((pred - train_Y)^2)
  if (mse < best.train.MSE) {
    best.train.MSE <- mse
    best.train.M <- M
    best.train.coefficients <- fit.pcr$coefficients[, , M]
  }
}
best.train.M #6
best.train.MSE #9593086
best.train.coefficients
# carat        cut      color    clarity      depth      table 
#2740.00485   47.62913 -367.05082  604.05949  -14.98773  -37.16489 


# prediction on test data
pred.test.list <- predict(fit.pcr, test_X)
best.test.M = 0.
best.test.MSE = Inf
best.test.coefficients = NULL
for (M in 1:ncol(train_X)) {
  pred = pred.test.list[, , M]
  mse = mean((pred - test_Y)^2)
  if (mse < best.test.MSE) {
    best.test.MSE <- mse
    best.test.M <- M
    best.test.coefficients <- fit.pcr$coefficients[, , M]
  }
}
best.test.M #6
best.test.MSE #9602035

########################################


# ridge regression

cv_ridge <- cv.glmnet(as.matrix(train_X), train_Y, alpha = 0)
best_lambda <- cv_ridge$lambda.1se
ridge_model <- glmnet(as.matrix(train_X), train_Y,
                      alpha = 0, lambda = best_lambda)
coef(ridge_model)

#(Intercept) 2999.473351
#carat       2417.869601
#cut           43.392618
#color       -250.570431
#clarity      442.393856
#depth         -9.379870
#table         -3.515143


# prediction on train data
ridge.train.pred <- predict(ridge_model, as.matrix(train_X))
ridge.train.MSE <- mean((ridge.train.pred - train_Y)^2)
ridge.train.MSE #677572.2

#prediction on test data
ridge.test.pred <- predict(ridge_model, as.matrix(test_X))
ridge.test.MSE <- mean((ridge.test.pred - test_Y)^2)
ridge.test.MSE #668929.1

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

# Calculating the root mean square error
postResample(predict_rf,test_Y)['RMSE']^2 # 140845.7

# Calculating the R^2 value
postResample(predict_rf,test_Y)['Rsquared'] # 0.97

plot(random_forest_model)

