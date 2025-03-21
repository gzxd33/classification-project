## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set up libraries
library(tidyverse)
library(caret)      
library(rpart)
library(rpart.plot)
library(randomForest) 
library(pROC) 

set.seed(307)
telecom <- read.csv("https://www.louisaslett.com/Courses/MISCADA/telecom.csv")
glimpse(telecom)
summary(telecom)
miss <- colSums(is.na(telecom))
if(sum(miss) > 0) {
  print("Missing value statistics:")
  print(miss[miss > 0])
  telecom$TotalCharges[is.na(telecom$TotalCharges)] <- telecom$MonthlyCharges[is.na(telecom$TotalCharges)]
}
telecom$Churn <- as.factor(telecom$Churn)
flow_tab <- table(telecom$Churn)
print("Churn:")
print(flow_tab)
if("Yes" %in% names(flow_tab)) {
  flow_pct <- flow_tab["Yes"] / sum(flow_tab) * 100
  cat("churn rate:", round(flow_pct, 2), "%\n")
} else if("True" %in% names(flow_tab)) {
  flow_pct <- flow_tab["True"] / sum(flow_tab) * 100
  cat("churn rate:", round(flow_pct, 2), "%\n")
} else if("1" %in% names(flow_tab)) {
  flow_pct <- flow_tab["1"] / sum(flow_tab) * 100
  cat("churn rate:", round(flow_pct, 2), "%\n")
} else {
  cat("error\n")
}

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
contract_viz <- telecom %>%
  group_by(Contract, Churn) %>%
  summarise(n = n(), .groups = 'drop') %>%
  group_by(Contract) %>%
  mutate(pct = n/sum(n)) %>%
  ggplot(aes(x = Contract, y = pct, fill = Churn)) +
  geom_col() +
  labs(title = "Contract Type and Churn") +
  theme_minimal()
fee_viz <- ggplot(telecom, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Monthly fee and turnover relationship") +
  theme_minimal()
churn_viz <- ggplot(telecom, aes(x = Churn, fill = Churn)) +
  geom_bar(aes(y = after_stat(count)/sum(after_stat(count)))) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Customer churn rate") +
  theme_minimal()

print(contract_viz)
print(fee_viz)
print(churn_viz)

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
svc <- c("PhoneService", "MultipleLines", "InternetService", 
         "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
         "TechSupport", "StreamingTV", "StreamingMovies")
telecom <- telecom %>%
  mutate(across(all_of(svc), 
                ~ifelse(. %in% c("Yes", "Fiber optic", "DSL"), 1, 0)))

telecom$svc_cnt <- rowSums(telecom[, svc])
ggplot(telecom, aes(x = Churn, y = svc_cnt, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Total number of services and churn") +
  theme_minimal()
fctr <- names(telecom)[sapply(telecom, is.factor)]
fctr <- fctr[fctr != "Churn"]
exists("telecom")
char_vars <- names(telecom)[sapply(telecom, is.character)]
telecom[char_vars] <- lapply(telecom[char_vars], as.factor)
fctr <- names(telecom)[sapply(telecom, is.factor)]
fctr <- fctr[fctr != "Churn"]
print(fctr)
telecom_factors <- telecom[, fctr, drop=FALSE]
enc <- dummyVars(" ~ .", data = telecom_factors)
telecom_bin <- predict(enc, newdata = telecom[, fctr])
telecom_bin <- as.data.frame(telecom_bin)


#enc <- dummyVars(" ~ .", data = telecom[, fctr])

#telecom_bin <- predict(enc, newdata = telecom[, fctr])
#telecom_bin <- as.data.frame(telecom_bin)
nmbr <- names(telecom)[sapply(telecom, is.numeric)]
telecom_prep <- cbind(
  telecom_bin,
  telecom[, nmbr],
  Churn = telecom$Churn
)

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4
idx <- createDataPartition(telecom_prep$Churn, p = 0.7, list = FALSE)
tr_set <- telecom_prep[idx, ]
ts_set <- telecom_prep[-idx, ]

cat("Training set:", nrow(tr_set), "Test set:", nrow(ts_set), "\n")


## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
kfold <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

evaluate <- function(model, data) {
  preds <- predict(model, data)
  prob_matrix <- predict(model, data, type = "prob")
  churn_levels <- levels(data$Churn)
  positive_class <- churn_levels[2]
  cat("Probability columns available:", paste(colnames(prob_matrix), collapse=", "), "\n")
  cat("Positive class identified as:", positive_class, "\n")

  probs <- prob_matrix[, positive_class]
  cm <- confusionMatrix(preds, data$Churn)
  roc_obj <- roc(data$Churn, probs)
  auc_val <- auc(roc_obj)
  list(
    cm = cm,
    accuracy = cm$overall["Accuracy"],
    precision = cm$byClass["Precision"],
    recall = cm$byClass["Sensitivity"],
    f1 = cm$byClass["F1"],
    auc = auc_val,
    positive_class = positive_class,
    roc = roc_obj
  )
}

# Logistic regression
lgr <- train(
  Churn ~ .,
  data = tr_set,
  method = "glm",
  trControl = kfold,
  metric = "ROC"
)

lgr_preds <- predict(lgr, ts_set)
lgr_cm <- confusionMatrix(lgr_preds, ts_set$Churn)
lgr_accuracy <- lgr_cm$overall["Accuracy"]
lgr_probs <- predict(lgr, ts_set, type = "prob")
pos_class <- levels(ts_set$Churn)[2]
lgr_roc <- roc(ts_set$Churn, lgr_probs[, pos_class])
lgr_auc <- auc(lgr_roc)
cat("Logistic regression - Accuracy:", round(lgr_accuracy, 3), 
    "AUC:", round(lgr_auc, 3), "\n")

# Random forest
rforest <- train(
  Churn ~ .,
  data = tr_set,
  method = "rf",
  trControl = kfold,
  metric = "ROC",
  tuneLength = 3
)
rf_preds <- predict(rforest, ts_set)
rf_cm <- confusionMatrix(rf_preds, ts_set$Churn)
rf_accuracy <- rf_cm$overall["Accuracy"]
rf_probs <- predict(rforest, ts_set, type = "prob")
pos_class <- levels(ts_set$Churn)[2]
rf_roc <- roc(ts_set$Churn, rf_probs[, pos_class])
rf_auc <- auc(rf_roc)

cat("Random forest - Accuracy:", round(rf_accuracy, 3), 
    "AUC:", round(rf_auc, 3), "\n")

# lda
lda_mdl <- train(
  Churn ~ ., 
  data = tr_set,
  method = "lda",
  trControl = kfold,
  metric = "ROC"
)
lda_results <- evaluate(lda_mdl, ts_set)
cat("LDA - Accuracy:", round(lda_results$accuracy, 3), 
    "AUC:", round(lda_results$auc, 3), "\n")

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mdls <- c("Logistic regression", "Random forest", "LDA")
accuracy <- c(lgr_accuracy, rf_accuracy, lda_results$accuracy)
auc <- c(lgr_auc, rf_auc, lda_results$auc)
f1 <- c(lgr_cm$byClass["F1"], rf_cm$byClass["F1"], lda_results$f1)

mdl_perf <- data.frame(
  Model = mdls,
  Accuracy = round(accuracy, 3),
  AUC = round(auc, 3),
  F1 = round(f1, 3)
)

print(mdl_perf)
plot(lgr_roc, col = "lightblue", main = "ROC Comparison Curve")
plot(rf_roc, col = "yellow", add = TRUE)
plot(lda_results$roc, col = "pink", add = TRUE)

legend("bottomright", 
       legend = c(
         paste("Logistic:", round(lgr_auc, 3)),
         paste("Random Forest:", round(rf_auc, 3)),
         paste("LDA:", round(lda_results$auc, 3))
       ),
       col = c("lightblue", "yellow", "pink"),
       lwd = 2)


## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
best_model <- lgr
best_probs <- lgr_probs[, pos_class]
tp_gain <- 800
fp_cost <- 100
fn_cost <- 600
cutoffs <- c(0.3, 0.4, 0.5, 0.6, 0.7)
biz_val <- data.frame(
  Threshold = cutoffs,
  NetValue = numeric(length(cutoffs))
)
pos_class <- levels(ts_set$Churn)[2]
neg_class <- levels(ts_set$Churn)[1]
for (i in 1:length(cutoffs)) {
  current_threshold <- cutoffs[i]
  preds <- factor(
    ifelse(best_probs >= current_threshold, pos_class, neg_class),
    levels = levels(ts_set$Churn)
  )
  cm <- table(preds, ts_set$Churn)
  tp <- cm[pos_class, pos_class]
  fp <- cm[pos_class, neg_class]
  fn <- cm[neg_class, pos_class]
  net_value <- (tp * tp_gain) - (fp * fp_cost) - (fn * fn_cost)
  biz_val$NetValue[i] <- net_value
}
best_idx <- which.max(biz_val$NetValue)
best_t <- biz_val$Threshold[best_idx]
cat("Best threshold:", best_t, "Net value:", biz_val$NetValue[best_idx], "\n")
barplot(biz_val$NetValue, names.arg = biz_val$Threshold, 
        main = "Different Thresholds",
        xlab = "Threshold", ylab = "Net Value")

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
y_pred <- factor(ifelse(best_probs >= best_t, pos_class, neg_class), 
                levels = levels(ts_set$Churn))
perf <- confusionMatrix(y_pred, ts_set$Churn)
saveRDS(best_model, "churn_model.rds")

