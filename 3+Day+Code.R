#Sensitivity Analysis, 3 Day Tap

setwd("H:/ML Ascites Tap and SBP 2023")
library(dplyr)

SBP_ML <- read.csv("3DAY.csv")
SecondAdmit <- read.csv("TestData_second.csv")

SBP_ML[SBP_ML=="NULL"] <- NA
SecondAdmit[SecondAdmit=="NULL"] <- NA

#Isolate matrix of predictors only
SBP_ML_Predictors <- SBP_ML[,-c(1:3)]
SecondAdmit_Predictors <- SecondAdmit[,-c(1,2,4)]

#Examine data missingness/zero proportion
Freqs <- apply(SBP_ML_Predictors, 2, function(i) mean(i == 0,na.rm=T))
Freqs[Freqs>0.999 & Freqs!=1]

NAs <- apply(SBP_ML_Predictors, 2, function(i) mean(is.na(i)))
sort(NAs, decreasing = T)[1:10]

drop_cols_na <- names(NAs[NAs>0.05])
drop_cols <- names(Freqs[Freqs>0.999 & Freqs!=1])

#Libraries
library(xgboost)
library(splitTools)
library(pROC)
library(caret)
library(randomForest)

SBP_ML_Clean <- data.frame(apply(SBP_ML_Predictors, 2, as.numeric))
SecondAdmit_Clean <- data.frame(apply(SecondAdmit_Predictors, 2, as.numeric))

SBP_ML_Clean$SBP <- as.factor(SBP_ML$SBP)
SecondAdmit_Clean$SBP <- as.factor(SecondAdmit$SBP_Second)

SBP_ML_Clean <- SBP_ML_Clean[,-which(colnames(SBP_ML_Clean) %in% c(drop_cols, "ActMI2yrs"))]
SecondAdmit_Clean <- SecondAdmit_Clean[,-which(colnames(SecondAdmit_Clean) %in% c(drop_cols, "ActMI2yrs"))]
SecondAdmit_Clean <- SecondAdmit_Clean[-which(SecondAdmit_Clean$SBP_First==1),]
SecondAdmit_Clean <- SecondAdmit_Clean[,-1]

#Train/test split 75% train 25% test
set.seed(2023)
inds <- partition(SBP_ML_Clean$SBP, p = c(train = 0.75, test = 0.25))

Train <- SBP_ML_Clean[inds$train,]
Test <- SBP_ML_Clean[inds$test,]

#hyperparameters tuned from gridsearch code
params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta=0.05,
               gamma=5,
               max_depth=6,
               min_child_weight=1,
               subsample=0.75,
               colsample_by_tree=1
)



#Model
set.seed(2023)
XGB <- xgboost(params=params,
               data=as.matrix(Train[,-ncol(Train)]),
               missing=NA,
               label=as.numeric(Train$SBP)-1,
               nrounds=100,
               eval_metric="auc"
)

Predictions <- predict(XGB, as.matrix(Test[,1:(ncol(Train)-1)]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.15,1,0), "True"=Test$SBP), positive = "1")

binom.test(185,185+7)
binom.test(1271,1271+101)
binom.test(1867,1867+180)

KeepTest <- c("TotalWBC","Platelets","AST"
              ,  "AlkPhos"     ,        "INR" ,                 "NeutrophilPercentage"
              ,    "Glucose"      ,       "BUN"    ,              "ALT"          
              ,  "AdmitTemp"      ,     "BMIAtIndex"   ,        "Albumin"            
              ,  "AdmitDiasBP"    ,     "Bilirubin"    ,        "EGFR"               
              ,  "EosinophilPercentage","Hemoglobin"  ,         "CO2"                
              ,  "AdmitSysBP", "Potassium")

set.seed(2023)
XGBoostTest <- xgboost(params=params,
                       data=as.matrix(Train[,which(colnames(Train) %in% KeepTest)]),
                       missing=NA,
                       label=as.numeric(Train$SBP)-1,
                       nrounds=100,
                       eval_metric="auc")

Predictions2 <- predict(XGBoostTest, as.matrix(Test[,which(colnames(Train) %in% KeepTest)]), type="response")

confusionMatrix(table("Prediction"=ifelse(Predictions2>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions2>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions2>0.15,1,0), "True"=Test$SBP), positive = "1")

binom.test(69,73)
binom.test(1120,1215)
binom.test(1791,1989)

Predictions_second <- predict(XGBoostTest, as.matrix(SecondAdmit_Clean[,which(colnames(Train) %in% KeepTest)]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.05,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.1,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.15,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")

binom.test(50,50)
binom.test(1071,1132)
binom.test(1817,1928)

Predictions_VCU <- predict(XGBoostTest, VCU_Features, type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU>0.05,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU>0.1,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU>0.15,1,0), "True"=VCU$SBP), positive = "1")

binom.test(4,4)
binom.test(84,86)
binom.test(141,145)

#Logistic Regression
KeepTest <- c("TotalWBC","Platelets","AST"
              ,  "AlkPhos"     ,        "INR"           
              ,    "Glucose"      ,       "BUN"    ,              "ALT"          
              ,  "AdmitTemp"      ,     "BMIAtIndex"   ,        "Albumin"            
              ,  "AdmitDiasBP"    ,     "Bilirubin"                 
              ,  "Hemoglobin"  ,         "CO2"                
              ,  "AdmitSysBP")

TrainReduced <- Train[,c(142,which(colnames(Train) %in% KeepTest))]
TestReduced <- Test[,c(142,which(colnames(Test) %in% KeepTest))]
SecondAdmit_Clean_Reduced <- SecondAdmit_Clean[,c(142,which(colnames(SecondAdmit_Clean) %in% KeepTest))]

LR_Model <- glm(data=TrainReduced, SBP ~., family=binomial(logit))
summary(LR_Model)

ROC_Red <- roc(response=Test$SBP, predictor=Predictions_sens2_valid1, percent=T,
               ci=T)
ROC_Red

Predictions_sens2_valid1 <- predict(LR_Model, TestReduced, "response")
Predictions_sens2_valid2 <- predict(LR_Model, SecondAdmit_Clean_Reduced, "response")
Predictions_sens2_valid3 <- predict(LR_Model, data.frame(VCU_Features[,-c(2,16,17,13,21)]), "response")

confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid1>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid1>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid1>0.15,1,0), "True"=Test$SBP), positive = "1")

binom.test(40,46)
binom.test(515,567)
binom.test(1146,1282)

confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.05,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.1,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.15,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")

binom.test(28,30)
binom.test(593,630)
binom.test(1121,1195)

confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.05,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.1,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.15,1,0), "True"=VCU$SBP), positive = "1")

binom.test(2,2)
binom.test(46,47)
binom.test(125,127)

