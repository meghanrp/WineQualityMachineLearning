########################################################
# Project: Wine Quality - Harvard Data Science Capstone
# Author: Meghan Patterson
########################################################

# Install/download the required packages/libraries if not already installed/downloaded
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(C50)) install.packages("C50", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(kernlab)
library(randomForest)
library(C50)
library(GGally)
library(rpart)
library(pROC)
library(psych)
library(e1071)
library(rpart.plot)

# Import the wine quality dataset
wine_data <- read.csv("https://raw.githubusercontent.com/meghanrp/Harvard-Data-Science-Capstone/master/winequalityN.csv", header = TRUE)

########################################
# Exploratory Data Analysis
########################################
# Obtain a summary of the dataset
summary(wine_data)

# Visualize the structure of the data
str(wine_data)

# Count the number of NAs in the dataset
sum(is.na(wine_data))
# Remove the NAs from the dataset - because there is not a significant number of NAs, removing them should not have a drastic effect on the results
wine_data <- na.omit(wine_data)

# Remove the "type" variable - the type variable is an identification of the type of wine used in the case and will not have an impact on the results
wine_data <- wine_data %>% select(-type)

# Create a new column that has two levels for the quality of wine
# Anything with a score of 5 or below is a 0
# Anything with a score above 5 is a 1 
wine_data <- wine_data %>% mutate(quality_new = as.factor(ifelse(quality <= 5, 0, 1))) %>% select(-quality)

# Visualize the new structure of the data
str(wine_data)

# Create a table to see the proportion of the two qualities of wine
prop.table(table(wine_data$quality_new))

# Create a boxplot to show the fixed acidity distribution for each quality level
gg_wine_acidity <- wine_data %>% ggplot(aes(quality_new, fixed.acidity, color = quality_new)) + geom_boxplot(outlier.color = "darkblue")
gg_wine_acidity

# Create a boxplot to show the volatile acidity distribution for each quality level
gg_volatile_acidity <- wine_data %>% ggplot(aes(quality_new, volatile.acidity, color = quality_new)) + geom_boxplot(outlier.color = "darkblue")
gg_volatile_acidity

# Create a boxplot to show the total sulfur dioxide distribution for each quality level
gg_total_sulfur <- wine_data %>% ggplot(aes(quality_new, total.sulfur.dioxide, color = quality_new)) + geom_boxplot(outlier.color = "darkblue")
gg_total_sulfur

# Create a boxplot to show the pH distribution for each quality level
gg_pH <- wine_data %>% ggplot(aes(quality_new, pH, color = quality_new)) + geom_boxplot(outlier.color = "darkblue")
gg_pH

# Create a boxplot to show the citric acid distribution for each quality level
gg_citric <- wine_data %>% ggplot(aes(quality_new, citric.acid, color = quality_new)) + geom_boxplot(outlier.color = "darkblue")
gg_citric

# Create a correlation plot with the predictors
cor_plot <- wine_data %>% ggcorr(method = c('complete.obs', 'pearson'), palette = "RdBu", breaks = 6, digits = 3, label = TRUE, label_color = "white")
cor_plot


# Set the seed
set.seed(1234)

# Split the datasets into two sets: one for the model creation/testing and a validation set for the final model test
split_index <- createDataPartition(wine_data$quality_new, p = 0.9, times = 1, list = FALSE)
model_data <- wine_data[split_index, ]
validation_set <- wine_data[-split_index, ]

# Split the model data subset into two datasets: one for training and one for testing 
train_index <- createDataPartition(model_data$quality_new, p = 0.9, times = 1, list = FALSE)
training_set <- model_data[train_index, ]
test_set <- model_data[-train_index, ]

# Check to see if proportions are similar for both the training set and testing set
prop.table(table(training_set$quality_new))
prop.table(table(test_set$quality_new))


#####################################
# Create a logistic regression model
#####################################
set.seed(1234)
# Create the model
glm_model <- train(quality_new ~ ., method = "glm", data = training_set)
summary(glm_model)

# Calculate the prediction
y_pred_glm <- predict(glm_model, test_set, type = "raw")

# Construct the confusion matrix
conf_glm <- confusionMatrix(y_pred_glm, test_set$quality_new)
conf_glm

# Add the accuracy results to a table
glm_acc <- conf_glm$overall[1]
accuracy_results <- data.frame(Model = "Logistic Regression", Accuracy = glm_acc)
accuracy_results



###########################
# Create a LDA model
############################
set.seed(1234)
# Create the model
lda_model <- train(quality_new~., method = "lda", data = training_set, trControl = trainControl(method = "cv"))

# Calculate the prediction
y_pred_lda <- predict(lda_model, test_set)

# Construct the confusion matrix
conf_lda <- confusionMatrix(y_pred_lda, test_set$quality_new)
conf_lda 

# Add the accuracy results to the existing table
lda_acc <- conf_lda$overall[1]
accuracy_results <- accuracy_results %>% add_row(Model = "Linear Discriminant Analysis (LDA)", Accuracy = lda_acc)
accuracy_results



############################
# Create a KNN model
############################
set.seed(1234)
# Construct the model
ctrl <- trainControl(method = "repeatedcv", repeats = 5)
knn_model <- train(quality_new~., method = "knn", data = training_set, trControl = ctrl)

# Create a plot to show the number of neighbors versus accuracy
plot(knn_model, print.three = 0.5, type = "S")

# Calculate the prediction
y_pred_knn <- predict(knn_model, test_set)

# Construct the confusion matrix
conf_knn <- confusionMatrix(y_pred_knn, test_set$quality_new)
conf_knn

# Add the accuracy results to the existing table
knn_acc <- conf_knn$overall[1]
accuracy_results <- accuracy_results %>% add_row(Model = "K-Nearest Neighbors (KNN)", Accuracy = knn_acc)
accuracy_results



##############################
# Create SVM model
###############################
set.seed(1234)
# Create a function to find the optimal cost value
Y <- NULL
k <- 30
for (i in 1:k) {
  svm.fit <- svm(quality_new~., training_set, scale = FALSE, kernel = "linear", cost = i)
  pred_svm <- predict(svm.fit, test_set)
  conf <- confusionMatrix(pred_svm, test_set$quality_new)
  Y[i] <- conf$overall[1]
}

# Find the cost value that gives the highest accuracy
plot(1:k, Y, pch = 20, xlab = "Cost", ylab = "Accuracy")
abline(v = which.max(Y))

# Create the model with a cost of 4
svm_fit <- svm(quality_new~., training_set, scale = FALSE, kernel = "linear", cost = 4)

# Calculate the prediction
svm_pred <- predict(svm_fit, test_set)

# Construct the confusion matrix
conf_svm <- confusionMatrix(svm_pred, test_set$quality_new)
conf_svm

# Add the results to the existing table
svm_acc <- conf_svm$overall[1]
accuracy_results <- accuracy_results %>% add_row(Model = "Support Vector Machine (SVM)", Accuracy = svm_acc)
accuracy_results



##############################
# Create a decision tree
##############################
set.seed(1234)
# Create the model
ctrl <- rpart.control(minsplit = 5L, maxdepth = 5L, minbucket = 5, cp = .002, maxsurrogate = 4)
dec_tree.model <- rpart(quality_new~., training_set, method = "class", control = ctrl)

# Calculate the prediction
y_pred_dec <- predict(dec_tree.model, test_set, type = "class")

# Construct the confusion matrix
conf_dec_tree <- confusionMatrix(y_pred_dec, test_set$quality_new)
conf_dec_tree

# Add results to the existing table
dec_acc <- conf_dec_tree$overall[1]
accuracy_results <- accuracy_results %>% add_row(Model = "Decision Tree", Accuracy = dec_acc)
accuracy_results

# Create the decision tree plot
prp(dec_tree.model)



###################################
# Create random forest model
##################################
set.seed(1234)
# Create the model
control <- trainControl(method = "cv", repeats = 5)
rf_model <- train(quality_new~., training_set, method = "rf", trControl = control)

# Calculate the prediction
y_pred_rf <- predict(rf_model, test_set)

# Create the confusion matrix
conf_rf <- confusionMatrix(y_pred_rf, test_set$quality_new)
conf_rf

# Add results to the existing table
rf_acc <- conf_rf$overall[1]
accuracy_results <- accuracy_results %>% add_row(Model = "Random Forest", Accuracy = rf_acc)
accuracy_results



####################################################################
# The random forest model is the model with the highest accuracy
# The validation dataset will be tested with this model
###################################################################
set.seed(1234)
# Create the model
control <- trainControl(method = "cv", repeats = 5)
rf_valid_model <- train(quality_new~., model_data, method = "rf", trControl = control)

# Calculate the prediction
y_pred_rf_valid <- predict(rf_valid_model, validation_set)

# Create the confusion matrix
conf_rf_valid <- confusionMatrix(y_pred_rf_valid, validation_set$quality_new)
conf_rf_valid

# Add results to the existing table
rf_valid_acc <- conf_rf_valid$overall[1]
accuracy_results <- accuracy_results %>% add_row(Model = "Random Forest - Validation", Accuracy = rf_valid_acc)
accuracy_results
