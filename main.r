library(tidyverse)
library(rpart)
library(dplyr)
library(rpart.plot)
library(ISLR2)
library(plyr)
library(ggplot2)
library(GGally)
library(leaps)

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

# subset selection'
fit.subset <- regsubsets(train_Y ~ ., data = train_X, method = "exhaustive")
coef(fit.subset, 1:5)
