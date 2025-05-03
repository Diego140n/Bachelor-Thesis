library(dplyr)
library(tidyr)
library(caret)
library(xgboost)
library(Matrix)
library(pROC)
library(ggplot2)
library(purrr)
library(patchwork)
library(randomForest)
library(MLmetrics)
library(reshape2)


# Load data
data <- read.csv("fraud_oracle.csv") 
data$FraudFound_P <- as.factor(data$FraudFound_P)
head(data)
dim(data)
str(data)
colnames(data)

data <- data %>%
  mutate(WitnessPresent = ifelse(WitnessPresent == "Yes", 1, 0))

data <- data %>%
  mutate(PoliceReportFiled = ifelse(PoliceReportFiled == "Yes", 1, 0))

data <- data %>%
  mutate(Sex = ifelse(Sex == "Male", 0, 1))

data <- data %>%
  mutate(MaritalStatus = ifelse(MaritalStatus == "Single", 0, 1))

data <- data %>%
  mutate(Fault = ifelse(Fault == "Third Party", 0, 1))

data <- data %>%
  select(-Age)

data <- data %>%
  select(-Year)

data <- data %>%
  select(-RepNumber)


#Check correlation
numeric_data <- data[, sapply(data, is.numeric)]
cor_matrix <- cor(numeric_data)

melted_cormat <- melt(cor_matrix)

# Create heat map
ggplot(melted_cormat, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()


# One hot encoding
factor_cols <- names(data)[sapply(data, function(x) is.factor(x) | is.character(x))]
factor_cols <- setdiff(factor_cols, "FraudFound_P")  # Exclude target variable

dummies <- dummyVars(~ ., data = data[, factor_cols, drop = FALSE])
data_encoded <- predict(dummies, newdata = data)

numeric_cols <- names(data)[sapply(data, is.numeric)]
data <- cbind(data[, numeric_cols, drop = FALSE], 
              data_encoded,
              FraudFound_P = data$FraudFound_P)


# Partition
set.seed(999)

train_index <- createDataPartition(data$FraudFound_P, p = 0.75, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


# Check outliers

true_numeric_cols <- setdiff(
  names(train_data)[sapply(train_data, is.numeric)],
  c("FraudFound_P", colnames(data_encoded))) 

numeric_data <- data %>% select(all_of(true_numeric_cols))

plot_list <- numeric_data %>%
  imap(~ ggplot(data, aes(y = .x)) +
         geom_boxplot(fill = "lightblue", outlier.color = "red") +
         labs(title = .y, y = "") +
         theme_minimal())

wrap_plots(plot_list, ncol = 3) 


# Standardize
preproc <- preProcess(train_data[, true_numeric_cols, drop = FALSE], 
                      method = c("center", "scale"))
trainData <- predict(preproc, train_data)
testData <- predict(preproc, test_data)

train_x <- trainData %>% select(-FraudFound_P) %>% as.matrix()
train_y <- as.numeric(trainData$FraudFound_P) - 1
test_x <- testData %>% select(-FraudFound_P) %>% as.matrix()
test_y <- as.numeric(testData$FraudFound_P) - 1


# Models

# RF
train_y <- as.factor(train_y)
rf_model <- randomForest(
  x = train_x,
  y = train_y,
  ntree = 500,
  strata = train_y,  # For stratified sampling
  sampsize = c("0" = sum(train_y == "0"), 
               "1" = sum(train_y == "1")),  # Balanced samples
  importance = TRUE
)

rf_pred <- predict(rf_model, test_x, type = "prob")[, "1"]
rf_pred_class <- ifelse(rf_pred > 0.3, "1", "0")

test_y <- as.factor(test_y)

conf_matrix_rf <- table(rf_pred_class, test_y)
conf_matrix_rf

TP <- conf_matrix_rf[2, 2]  
FN <- conf_matrix_rf[1, 2] 
FP <- conf_matrix_rf[2, 1]

recall <- TP / (TP + FN)
recall

precision <- TP / (TP + FP)
precision

type2 <- FN / (TP + FN)
type2

F1_score <- 2*(recall*precision)/(recall + precision)
F1_score



###---###

# Prepare data matrices

train_x <- trainData %>% select(-FraudFound_P) %>% as.matrix()
train_y <- as.numeric(trainData$FraudFound_P) - 1
test_x <- testData %>% select(-FraudFound_P) %>% as.matrix()
test_y <- as.numeric(testData$FraudFound_P) - 1

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

# class imbalance
neg <- sum(train_y == 0)
pos <- sum(train_y == 1)
scale_pos_weight <- neg / pos
scale_pos_weight

# XGBoost Model
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = scale_pos_weight,
  eta = 0.1,
  max_depth = 6
)

xgb_model <- xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

xgb_probs <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_probs > 0.3, 1, 0)
xgb_pred <- factor(xgb_pred, levels = c(0, 1))
test_y_factor <- factor(test_y, levels = c(0, 1))

conf_matrix_xg <- table(xgb_pred, test_y_factor)
conf_matrix_xg

TP <- conf_matrix_xg[2, 2]  
FN <- conf_matrix_xg[1, 2] 
FP <- conf_matrix_xg[2, 1]

recall <- TP / (TP + FN)
recall

precision <- TP / (TP + FP)
precision

type2 <- FN / (TP + FN)
type2

F1_score <- 2*(recall*precision)/(recall + precision)
F1_score


# XGBoost ROC
xgb_roc <- roc(test_y, xgb_probs)
plot(xgb_roc, col = "blue", main = "ROC Curves")
auc(xgb_roc)

# Random Forest ROC
rf_roc <- roc(as.numeric(test_y) - 1, rf_pred)
lines(rf_roc, col = "green")
auc(rf_roc)

