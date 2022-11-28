library(tidyverse)
library(rpart)
library(rpart.plot)
library(ISLR2)
library(plyr)
library(ggplot2)
library(GGally)

diamonds
#mean = mean(diamonds$cut)
carat_mean = mean(diamonds$carat)
carat_sd = sd(diamonds$carat)

carat_Tmin = carat_mean-(3*carat_sd)
carat_Tmax = carat_mean+(3*carat_sd)

diamonds$carat[which(diamonds$carat < carat_Tmin | diamonds$carat > carat_Tmax)]

diamonds$cut = revalue(diamonds$cut, c("Ideal"="1","Premium"=2,"Good"="3","Very Good"="4","Fair"="5"))
diamonds$cut = unclass(diamonds$cut)
diamonds$clarity = unclass(diamonds$clarity)
diamonds$color = unclass(diamonds$color)
diamonds
hist()
ggpairs(diamonds)
par(mfrow=c(1,3))
diamonds %>%
  ggplot(aes(price,carat,col=clarity))+
  geom_point()
diamonds %>%
  ggplot(aes(price,carat,col=cut))+
  geom_point()
diamonds %>%
  ggplot(aes(price,carat,col=color))+
  geom_point()

