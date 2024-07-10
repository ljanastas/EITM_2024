install.packages("caret")
install.packages("mlbench")
install.packages("randomForest")

# Load the necessary libraries
library(caret)
library(mlbench)
library(randomForest)

# Load the PimaIndiansDiabetes dataset
data(PimaIndiansDiabetes)

# Set the seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
trainIndex <- createDataPartition(PimaIndiansDiabetes$diabetes, p = 0.8, list = FALSE)
trainData <- PimaIndiansDiabetes[trainIndex, ]
testData <- PimaIndiansDiabetes[-trainIndex, ]

# Define the training control
trainControl <- trainControl(method = "cv", number = 10)

# Train the model using the random forest method
model <- train(diabetes ~ ., data = trainData, method = "rf", trControl = trainControl)

# Print the model details
print(model)

# Make predictions on the test data
predictions <- predict(model, newdata = testData)

# Evaluate the model performance
confMatrix <- confusionMatrix(predictions, testData$diabetes,positive = "pos")
print(confMatrix)

### Variable Importance
# Get variable importance from the model
importance <- varImp(model, scale = FALSE)

# Print the variable importance
print(importance)

# Plot variable importance using ggplot2
plot <- ggplot(importance, aes(x = reorder(Overall, desc(Overall)), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Variable") +
  ylab("Importance") +
  ggtitle("Variable Importance Plot")

print(plot)
