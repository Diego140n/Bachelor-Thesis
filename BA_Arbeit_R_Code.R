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
library(SHAPforxgboost)


# Load data
data <- read.csv("fraud_oracle.csv") 
data$FraudFound_P <- as.factor(data$FraudFound_P)
dim(data)

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
  c("FraudFound_P", colnames(data_encoded), "WitnessPresent", "PoliceReportFiled", "Sex")
)

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

# Classification Threshold

roc_obj <- roc(test_y, rf_pred)
opt_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
opt_coords

# Confusion matrix and metrics

rf_pred_class <- ifelse(rf_pred > 0.05, "1", "0")

test_y <- as.factor(test_y)

conf_matrix_rf <- table(rf_pred_class, test_y)
conf_matrix_rf

TP <- conf_matrix_rf[2, 2]  
FN <- conf_matrix_rf[1, 2] 
FP <- conf_matrix_rf[2, 1]
TN <- conf_matrix_rf[1, 1]

Accuracy <- (TP + TN)/(TP + TN + FP + FN)
Accuracy

recall <- TP / (TP + FN)
recall

precision <- TP / (TP + FP)
precision

type2 <- FN / (TP + FN)
type2

F1_score <- 2*(recall*precision)/(recall + precision)
F1_score



### Cross validation RF

class_counts <- table(trainData$FraudFound_P)
min_class_size <- min(class_counts)


class_weights <- as.vector(1 / class_counts)
names(class_weights) <- names(class_counts)

trainData$FraudFound_P <- factor(ifelse(trainData$FraudFound_P == 1, "Yes", "No"))

set.seed(999)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

rf_cv_model <- train(
  FraudFound_P ~ .,
  data = trainData,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",  # AUC optimization
  preProcess = c("center", "scale"),
  
  tuneLength = 3
)

print(rf_cv_model)


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

# Classification Treshold
roc_obj <- roc(test_y, xgb_probs)
opt_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
opt_coords

# Conf. Matrix and Metrics
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

# Calibration Analysis
binary_y <- ifelse(test_y == "Yes", 1, 0)
# Manual Brier Score implementation
brier_score <- function(y_true, y_prob) {
  mean((y_true - y_prob)^2)
}

brier_rf <- brier_score(test_y, rf_pred)
brier_xgb <- brier_score(test_y, xgb_probs)
brier_rf
brier_xgb

# Feature importance

shap_values <- shap.values(xgb_model = xgb_model, X_train = test_x)
shap_values$mean_shap_score[3]



### Cross-Validation XGB

set.seed(999)
folds <- createFolds(train_y, k = 5, list = TRUE)

auc_list <- c()
f1_list <- c()

for (i in seq_along(folds)) {
  val_idx <- folds[[i]]
  
  dtrain_cv <- xgb.DMatrix(data = train_x[-val_idx, ], label = train_y[-val_idx])
  dval_cv <- xgb.DMatrix(data = train_x[val_idx, ], label = train_y[val_idx])
  
  model_cv <- xgb.train(
    params = params,
    data = dtrain_cv,
    nrounds = 200,
    watchlist = list(val = dval_cv),
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  preds <- predict(model_cv, dval_cv)
  pred_labels <- ifelse(preds > 0.3, 1, 0)
  
  # Evaluation
  roc_obj <- roc(train_y[val_idx], preds)
  auc_fold <- auc(roc_obj)
  
  cm <- table(factor(pred_labels, levels = c(0, 1)),
              factor(train_y[val_idx], levels = c(0, 1)))
  
  TP <- cm[2, 2]
  FP <- cm[2, 1]
  FN <- cm[1, 2]
  
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1 <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
  
  auc_list <- c(auc_list, auc_fold)
  f1_list <- c(f1_list, f1)
}

cat("XGBoost 5-Fold CV Mean AUC:", round(mean(auc_list), 4), "\n")
cat("XGBoost 5-Fold CV Mean F1 Score:", round(mean(f1_list), 4), "\n")
cat("XGBoost 5-Fold CV Mean recall:", round(mean(recall), 4), "\n")
cat("XGBoost 5-Fold CV Mean precision:", round(mean(precision), 4), "\n")




### ROC
xgb_roc <- roc(test_y, xgb_probs, ci = TRUE)
rf_roc <- roc(test_y, rf_pred, ci = TRUE)

xgb_auc <- auc(xgb_roc)
xgb_ci <- ci.auc(xgb_roc)
rf_auc <- auc(rf_roc)
rf_ci <- ci.auc(rf_roc)

plot(xgb_roc,
     col = "blue", lwd = 2,
     main = "ROC Curve: XGBoost vs. Random Forest",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     print.auc = FALSE)

lines(rf_roc, col = "forestgreen", lwd = 2)

abline(a = 0, b = 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c(
         paste0("XGBoost: AUC = ", round(xgb_auc, 3),
                " [", round(xgb_ci[1], 3), ", ", round(xgb_ci[3], 3), "]"),
         paste0("Random Forest: AUC = ", round(rf_auc, 3),
                " [", round(rf_ci[1], 3), ", ", round(rf_ci[3], 3), "]")
       ),
       col = c("blue", "forestgreen"),
       lwd = 2, cex = 0.8, bty = "n")
