library(ggplot2)
library(gridExtra)
library(caret)
library(rpart)
library(ggraph)
library(igraph)
library(dplyr)
library(rattle)
#Setting the random seed for reproducibility
set.seed(1023)

#In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
#They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
#More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har 
#(see the section on the Weight Lifting Exercise Dataset).

#We can start by looking at the information provided from test and training data and go through with some EDA
#Head and tail of the information as-is is shown below
head(pml_training)
tail(pml_training)

names(pml_training)
#A summaary of the dataset is provided below
summary(pml_training)
#We have information for 6 users: Carlitos, Pedro, Adelmo, Charles, Eurico and Jeremy
unique(pml_training$user_name)
#The classes we need to predict are stored in the 'classe' column
unique(pml_training$classe)



#the sapply performed on the columns removes the messy information we want to remove, no further information is provided by the excercise
#instructions. We can  assume that all NA values can be safely removed in order to clean the training data, we also remove information that seems 
#not important and unnecessary raw information that is present.

dat <- pml_training
dat <- dat[,sapply(dat,function(x) !any(is.na(x)))]

user <- dat$user_name
rest <- dat[,8:60]
final_training_data <- cbind(user,rest)
g2 <-  ggplot(final_training_data, aes(x = classe)) + geom_histogram(stat='count')
g2

#We need to do the same thing for the test data, the procedure is the same.  
testing <- pml_testing
testing <- testing[,sapply(testing,function(x) !any(is.na(x)))]
user <-  testing$user_name
b <- testing[,8:60]
final_testing_data <- cbind(user,b)


#The strategy adopted here is to create a data partition between a training and evaluation set (At this stage we are still working with the training set provided).
#The training set is cross-validated in 5 folds to measure our in-sample performance. The trained model is then used on the evalutation (training) partition to predict
#the values we can check for the out-of-sample performance
part <- createDataPartition(final_training_data$classe,p = 0.8, list = FALSE)
final_training_data = final_training_data[part,]
final_evaluation_data = final_training_data[-part,]
train_control<- trainControl(method="cv", number=5, savePredictions = TRUE)

#I tried implementing three different models, with increasing complexity, and as we will see, with increasing accuracy.
#The first model is the multinomial logistic regression, that seemed a great place to benchmark the following models

# Multinomial logistic regression
multi.logistic <- train(classe ~ ., data=final_training_data, method="multinom",
                  trControl=train_control)

#the model information and confusion matrix is shown below for the in-sample accuracy
multi.logistic$metric
confusionMatrix(multi.logistic)


#The predictions and out-of-sample accuracy is shown below 
pred.multi.logistic <- predict(multi.logistic,final_evaluation_data)
factor.evaluation <- as.factor(final_evaluation_data$classe)
confusionMatrix(pred.multi.logistic, factor.evaluation)

#As we can see the accuracy isn't great. Nonetheless, it's a great starting point, any increase in accuracy can be benchmarked with this model
#We can therefore reference back to it if things don't turn out great as we increase the complexity


# Recursive partitioning algorithm is the second model that was tested on the data.
#Briefly, Recursive partitioning is a statistical method used for multivariable analysis.
#The algorithm creates a decision tree used to correctly classify members 
#of the population by splitting it into sub-populations based on several dichotomous independent variables.

decision.tree <- train(classe ~ ., data=final_training_data, method = 'rpart', trControl = train_control)

decision.tree

            #In sample
confusionMatrix(decision.tree)
fancyRpartPlot(decision.tree$finalModel)

          #Out of sample 
pred.decision.tree <- predict(decision.tree,final_evaluation_data)
confusionMatrix(pred.decision.tree, factor.evaluation)
fancyRpartPlot(decision.tree$finalModel)

#As we can see, the accuracy is even worst than the previous model. Hopefully, the next model will yield better results.

#The final model used is Random Forest, which is well-known for its high accuracy at the expense of interpretability.


rf.model <- train(classe ~ ., data=final_training_data, method="rf",
                  trControl=train_control)


#The in-sample accuracy is incredibly high (an average of 99.24%), we will now check with the evaluation set 
confusionMatrix(rf.model)

predictions <- predict(rf.model,final_evaluation_data)
a <- as.factor(final_evaluation_data$classe)
confusionMatrix(predictions,a)
plot(rf.model$finalModel)

#The accuracy is very high with the evaluation set aswell, we can proceed by picking this model for our test data. Hopefully, the predictions on the 
#test set are going be accurate aswell

predictions_test <- predict(rf.model, final_testing_data)
predictions_test
#The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. 
#If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.
