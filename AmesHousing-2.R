# Predicting Housing Prices
# Ethan Nusbaum
# Data obtained from Kaggle https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# Ames Housing Dataset
library(corrplot)
library(gbm)
library(glmnet)
library(randomForest)
library(keras)

# Read in test and train data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
test$SalePrice <- 0
train$is.test <- 0
test$is.test <- 1
data <- rbind(train,test)
str(train) # 81 variables and 1460 observations SalePrice is the variable we want to predict
hist(train$SalePrice)
# This has skew, lets take log to normalize
data$logSalePrice <- log(data$SalePrice + 1)
hist(data$logSalePrice[data$is.test==0])
summary(train)
# MSSubClass should be a factor as it identifies a type of dwelling.
data$MSSubClass <- as.factor(data$MSSubClass)

# Overall Quality and Overall Condition are ordinal
data$OverallCond <- as.ordered(data$OverallCond)
data$OverallQual <- as.ordered(data$OverallQual)

################ Data Exploration and Feature Engineering ##########################
numerics <- sapply(data, is.numeric)
categoricals <- !numerics
correlation <- cor(na.omit(data[,numerics]))
corrplot(correlation)
# We see stong correlation between sales price and a few variables that we should look at:
# Year, Remodel Year, Bsmnt SF, 1st Flr SqFt, Living Area, Full Bath, Garage


# Finding NA values
na.count <- apply(data, 2, function(x)  sum(is.na(x)))
na.count <- data.frame(na.count)/dim(data)[1]
prcnt.na.count <- na.count*100
names(prcnt.na.count) <- "%_N/A"
par(mar = c(8,4,2,2))
barplot(prcnt.na.count$"%_N/A", names.arg = rownames(prcnt.na.count), las = 2, main = "% of N/A", ylim = c(0,100))
abline(h = 40, col = "red", lty = 2)
large.na <- row.names(na.count)[na.count>=0.4]
# We see Alley, Fireplace Quality, Pool, Fence, and Misc Feature all have large % of NA values
summary(data[,names(data) %in% large.na])
par(mfrow = c(ceiling(length(large.na)/2), 2))
for (i in 1:length(large.na)) {
  barplot(summary(data[,large.na[i]]), main = large.na[i])
}
par(mfrow = c(1,1))


# We do not want to just get rid of these values, so lets Replace These N/A values with "None" to avoid issues later
for (i in 1:length(categoricals)) {
  if (categoricals[i]) {
    if (!("None" %in% levels(data[,i]))) {
      levels2 <- levels(data[,names(categoricals)[i]])
      levels2 <- c(levels2,"None")
      data[,names(categoricals)[i]] <- factor(data[,names(categoricals)[i]], levels = levels2)
      data[,names(categoricals)[i]][is.na(data[,names(categoricals)[i]])] <- "None"
    }
  }
}



# Lets look at how many houses have been remodeled
data$Remodel <- as.factor(data$YearBuilt != data$YearRemodAdd)
levels(data$Remodel) <- c("No","Yes")
barplot(summary(data$Remodel), main = "Remodeled")
table(data$Remodel)

"
  # Looking at Correlations
  highCor <- names(which(correlation[,"SalePrice"]>0.5))
  par(mfrow = c(4,3))
  for (i in 1:length(highCor)) {
    plot(data[,highCor[i]], data$SalePrice, xlab = highCor[i], ylab = "Sales Price")
    abline(lm(data$SalePrice~data[,highCor[i]]), col="red")
  }
  par(mfrow = c(1,1))
"

# Need to check for Missing Values
length(which(!complete.cases(data)))
# We hve 339 observations with missing values
na.count2 <- apply(data, 2, function(x)  sum(is.na(x)))
na.count2 <- data.frame(na.count2)/dim(train)[1]
prcnt.na.count2 <- na.count2*100
names(prcnt.na.count2) <- "%_N/A"
par(mar = c(8,4,2,2))
barplot(prcnt.na.count2$"%_N/A", names.arg = rownames(prcnt.na.count2), las = 2, main = "% of N/A", ylim = c(0,100))
# The Missing values are in LotFrontage, GarageYrBuilt, and MasVnrArea, MasVnrType
na.left <- row.names(na.count2)[na.count2>0]
# Any MasVnrType that is N/A we are unsure of what it is and will throw these 8 observations out, becuase there is already a None type
data <- data[-which(is.na(data$MasVnrType)),]
summary(data[,names(data) %in% na.left])
# This has fixed the MasVnrAre variable also, so now we only have LotFrontage and GarageYrBuilt along with a few one offs on basements
# We will assume that if LotFrontage is N/A then there is no street connected to the property
data[which(is.na(data$LotFrontage)),"LotFrontage"] <- 0
summary(data[,names(data) %in% na.left])

# For any Houses without garages we will change the GarageYrBuilt value to the YearBuilt value
no.garage <- which(data$GarageType=="None")
for (i in 1:length(no.garage)) {
  data[no.garage[i],"GarageYrBlt"] <- data[no.garage[i],"YearBuilt"]
}
summary(data[,names(data) %in% na.left])

# We will eliminate the few observations left with N/A
data <- data[complete.cases(data),]
summary(data[,names(data) %in% na.left])

# Check that all categorical variables have more than one level

cat.names <- names(data[,categoricals])
problems <- ""
for (i in 1:length(cat.names)) {
  if (length(levels(data[,cat.names[i]]))==1) {
    problems <- c(problems,cat.names[i])
  }
}
if (length(problems) > 1) {
  problems[2:length(problems)]
}


####################################  Starting Regression  ###########################################
# Need to eliminate the SalePrice column as we will regress on log(SalePrice) and ID column becuase it will not help us
train.reg <- data[data$is.test == 0,-c(1,81,82)]
data.reg <- data[,-c(1,81)]

# First we will split our train data set into test and train for a LASSO Regression woth 10% left for testing the one-hot encode
test.subset <- sample(nrow(train.reg),nrow(train.reg)/10)
train.subset <- -test.subset

data.reg.model <- model.matrix(logSalePrice~., data = data.reg)[,-1]
train.rows <- which(data.reg.model[,"is.test"] == 0)
train.reg.model <- data.reg.model[train.rows,]
test.reg.model <- data.reg.model[-train.rows,]
train.reg.model <-subset(train.reg.model, select = -is.test)
test.reg.model <-subset(test.reg.model, select = -is.test)
y <- train.reg$logSalePrice

# Linear Regression With all Variables

lm.all <- lm(logSalePrice~.,data = train.reg)
summary(lm.all)

# Lasson Regression to Find Most Important Variables

fit.L <- glmnet(train.reg.model[train.subset,], y[train.subset], alpha = 1, family = "gaussian")
CV.L <- cv.glmnet(train.reg.model[train.subset,],y[train.subset],alpha=1, family = "gaussian") # alpha = 1 for lasso
plot(CV.L)  # Draws lines at the minimum CV and the 1 SE away from min CV lambda values. These are the ones we want! Each dot is 10 regressions (leaving different variable out each time) and find errors using OOS test then calculate MSE
L.min <- CV.L$lambda.min # Lambda with minimum MSE
L.1se <- CV.L$lambda.1se # Lambda within 1 SE of min MSE
CV.L$cvm[which(CV.L$lambda==c(L.1se))] # We check the CV to see if the simpler model is acceptable
CV.L$cvm[which(CV.L$lambda==c(L.min))] # 1se CV is very close to min, so we will use the 1se

# MSE
PR.LM <- predict(fit.L, s=L.min, newx=train.reg.model[test.subset,])
sqrt(mean((exp(PR.LM) - exp(y[test.subset]))^2))
# Error of ~ 21k
plot(exp(PR.LM),exp(y[test.subset]))

PR.L1 <- predict(fit.L, s=L.1se, newx=train.reg.model[test.subset,])
sqrt(mean((exp(PR.L1) - exp(y[test.subset]))^2))
# Error of ~25k
plot(exp(PR.L1),exp(y[test.subset]))


# Random Forest for prediction

# Lets test multiple numbers of trees to see error differences
error.tree <- sqrt(mean((exp(PR.LM) - exp(y[test.subset]))^2))
for (i in seq(1,301,50)) {
  housing.rf = randomForest(train.reg.model[train.subset,],y[train.subset], ntree=i) # Due to constraints of my computer limiting the number of trees
  housing.rf.price = predict(housing.rf, train.reg.model[test.subset,])
  error.tree <- c(error.tree,sqrt(mean((exp(y[test.subset])-exp(housing.rf.price))^2)))
}
error.tree <- error.tree[-1]
plot(seq(1,301,50),error.tree, ylab = "Mean Error", xlab = "Number of Trees in Forest", type = "l")
min(error.tree)
# Minnimum error of about $20k with about 150 trees in forest


# ANN

# Standardize Features
mean <- colMeans(train.reg.model)
stdev <- apply(train.reg.model, 2, sd)
train.ann <- scale(train.reg.model, center = mean, scale = stdev)
zero_var <- which(colSums(is.na(train.ann)) > 0, useNames = FALSE)
test.ann <- train.ann[test.subset,]
train.ann <- train.ann[-test.subset,]
train.ann <- train.ann[, -zero_var]
test.ann  <- test.ann[, -zero_var]

model <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = "relu", input_shape = ncol(train.ann)) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1) %>%
  
  # backpropagation
  compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )

learn <- model %>% fit(
  x = train.ann,
  y = y,
  epochs = 25,
  batch_size = 32,
  validation_split = .2,
  verbose = FALSE
)

learn
plot(learn)