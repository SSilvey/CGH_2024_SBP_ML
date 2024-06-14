library(dplyr)

#Read in the data / cleanings
SBP_ML <- read.csv("TestData.csv")
SecondAdmit <- read.csv("TestData_second.csv")

SBP_ML_Gender <- read.csv("TestData_Gender.csv")

Gender <- NULL
for (i in 1:nrow(SBP_ML)) {
  Gender <- append(Gender, SBP_ML_Gender[which(SBP_ML_Gender$PatientICN == SBP_ML$PatientICN[i]), "Gender"])
}

Gender[Gender=="NULL"] <- NA

SBP_ML$Gender <- ifelse(Gender=="M", 1, 0)

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

SBP_ML_Clean$SBP <- as.factor(SBP_ML$SBP_Outcome)
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

#XGBoost
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

#Plot Importance
xgb.plot.importance(xgb.importance(colnames(Train[,1:(ncol(Train)-1)]),model=XGB)[1:50,])
importances <- xgb.importance(colnames(Train[,1:(ncol(Train)-1)]),model=XGB)

Predictions <- predict(XGB, as.matrix(Test[,1:(ncol(Train)-1)]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.5,1,0), "True"=Test$SBP), positive = "1")

#ROC Curve
ROC_Full <- roc(response=Test$SBP, predictor=Predictions, percent=T,
                ci=T)

plot.roc(ROC_Full, print.auc = T, )

#Predictions (Test Set, Full Model)
confusionMatrix(table("Prediction"=ifelse(Predictions>0.5,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions>0.15,1,0), "True"=Test$SBP), positive = "1")

#CIs for NPV - full model
binom.test(226,236)
binom.test(1066,1066+72)
binom.test(1577,1577+132)

#Top variables 
Keep <- importances[1:30,"Feature"]
Keep

Keep$Feature #True top 20
#Remove PTT, Lipase, MaxTemp, PT and replace with next 4 (exclusing maxDiasBP)

#Univariate on top 20

FeatureUnivariate <- NULL

for (i in 1:30) {
  varName <- Keep$Feature[i]
  NoSBP <- paste(round(median(SBP_ML_Clean[SBP_ML_Clean$SBP==0, which(colnames(SBP_ML_Clean) %in% varName)],na.rm=T),5), "(",unname(round(quantile(SBP_ML_Clean[SBP_ML_Clean$SBP==0, which(colnames(SBP_ML_Clean) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(SBP_ML_Clean[SBP_ML_Clean$SBP==0, which(colnames(SBP_ML_Clean) %in% varName)],na.rm=T)[4],5)),")")
  SBP <- paste(round(median(SBP_ML_Clean[SBP_ML_Clean$SBP==1, which(colnames(SBP_ML_Clean) %in% varName)],na.rm=T),5), "(",unname(round(quantile(SBP_ML_Clean[SBP_ML_Clean$SBP==1, which(colnames(SBP_ML_Clean) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(SBP_ML_Clean[SBP_ML_Clean$SBP==1, which(colnames(SBP_ML_Clean) %in% varName)],na.rm=T)[4],5)),")")
  pvalue <- wilcox.test(SBP_ML_Clean[,which(colnames(SBP_ML_Clean) %in% varName)] ~ SBP_ML_Clean$SBP)$p.value
  
  vector <- c("Feature"=varName, 
              "Rank (overall)" = which(importances$Feature %in% varName),"No SBP" = NoSBP, "SBP" = SBP, "p-value" = ifelse(pvalue<0.001, "<0.001",pvalue))
  FeatureUnivariate <- data.frame(rbind(FeatureUnivariate, vector))
}

View(FeatureUnivariate)

PredictDF <- data.frame(Test$SBP, Predictions)

#Top 20 only
KeepTest <- importances[,"Feature"]
set.seed(2023)
XGBoostTest <- xgboost(params=params,
                        data=as.matrix(Train[,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24)]])]),
                        missing=NA,
                        label=as.numeric(Train$SBP)-1,
                        nrounds=100,
                        eval_metric="auc")

Predictions2 <- predict(XGBoostTest, as.matrix(Test[,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24)]])]), type="response")

confusionMatrix(table("Prediction"=ifelse(Predictions2>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions2>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions2>0.15,1,0), "True"=Test$SBP), positive = "1")

#NPVs for Reduced Top 20 Model 
binom.test(164,170)
binom.test(1035,1113)
binom.test(1507,1645)

#ROC Curve Top-20 Model
ROC_Red <- roc(response=Test$SBP, predictor=Predictions2, percent=T,
                ci=T)

plot.roc(ROC_Red, print.auc = T, )

plot(ROC_Full, print.auc = T, col="red", xaxt="n")
plot(ROC_Red, print.auc=T, col="Blue",
     print.auc.y=39, add=T)
roc.test(ROC_Full, ROC_Red)
axis(1, at=seq(0,100,20), pos = -4)

#Second paracentesis predictions
Predictions_second <- predict(XGBoostTest, as.matrix(SecondAdmit_Clean[,which(colnames(Train) %in% FeatureUnivariate$Feature[c(1:25)[-c(2,8,10,15,24)]])]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.05,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.5,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.15,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_second>0.1,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")

#NPVs / CI for Second Paracentesis Set
binom.test(168,170)
binom.test(1202,1261)
binom.test(1840,1947)

#EXPORT MODEL
#xgb.save(XGBoostTest, fname="XGBModel")

predict(XGBoostTest, newdata=as.matrix(SecondAdmit_Clean[1,which(colnames(Train) %in% FeatureUnivariate$Feature[c(1:25)[-c(2,8,10,15,24)]])]))
XGBoostTest$feature_names

FeatureUnivariate$Feature[c(1:25)[-c(2,8,10,15,24)]]

library(dcurves)

#SENSITIVITY ANALYSES
#1.) Dropping high-missingness variables
#Dropping all w/missingness over 10%
#Eosiniphils, Neutrophils, EGFR, Potassium
XGBoostTest$feature_names

set.seed(2023)
XGBoostTest2 <- xgboost(params=params,
                       data=as.matrix(Train[,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24,7,20,19,25)]])]),
                       missing=NA,
                       label=as.numeric(Train$SBP)-1,
                       nrounds=100,
                       eval_metric="auc")

XGBoostTest2$feature_names

Predictions_sens1_valid1 <- predict(XGBoostTest2, as.matrix(Test[,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24,7,20,19,25)]])]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_valid1>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_valid1>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_valid1>0.15,1,0), "True"=Test$SBP), positive = "1")

binom.test(129,132)
binom.test(1022,1022+79)
binom.test(1509,1509+146)

Predictions_sens1_valid2 <- predict(XGBoostTest2, as.matrix(SecondAdmit_Clean[,which(colnames(SecondAdmit_Clean) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24,7,20,19,25)]])]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_valid2>0.05,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_valid2>0.1,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_valid2>0.15,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")

Predictions_sens1_external <- predict(XGBoostTest2, VCU_Features[,-c(2,16,17,13,21)], type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_external>0.05,1,0), "True"=VCU_Response[,21]), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_external>0.1,1,0), "True"=VCU_Response[,21]), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens1_external>0.15,1,0), "True"=VCU_Response[,21]), positive = "1")

ROC_Red <- roc(response=Test$SBP, predictor=Predictions_sens1_valid1, percent=T,
               ci=T)
ROC_Red


#2.) Logistic Regression with Same Variables

TrainReduced <- Train[,c(143,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24,7,20,19,25)]]))]
TestReduced <- Test[,c(143,which(colnames(Test) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24,7,20,19,25)]]))]
SecondAdmit_Clean_Reduced <- SecondAdmit_Clean[,c(142,which(colnames(SecondAdmit_Clean) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,15,24,7,20,19,25)]]))]

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
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid1>0.5,1,0), "True"=Test$SBP), positive = "1")

binom.test(73,77)
binom.test(578,578+52)
binom.test(993,993+115)

confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.05,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.1,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.15,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid2>0.5,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")

binom.test(55,58)
binom.test(593,630)
binom.test(1121,1195)

confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.05,1,0), "True"=VCU_Response[,21]), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.1,1,0), "True"=VCU_Response[,21]), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.15,1,0), "True"=VCU_Response[,21]), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_sens2_valid3>0.5,1,0), "True"=VCU_Response[,21]), positive = "1")

binom.test(6,6)
binom.test(63,64)
binom.test(125,127)

#3 Dropping BMI

XGBoostTest_noBMI <- xgboost(params=params,
                             data=as.matrix(Train[,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,14,15,24)]])]),
                             missing=NA,
                             label=as.numeric(Train$SBP)-1,
                             nrounds=100,
                             eval_metric="auc")

Predictions_noBMI_testset <- predict(XGBoostTest_noBMI, as.matrix(Test[,which(colnames(Train) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,14,15,24)]])]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_noBMI_testset>0.05,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_noBMI_testset>0.1,1,0), "True"=Test$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_noBMI_testset>0.15,1,0), "True"=Test$SBP), positive = "1")

#NPVs / CIs
binom.test(160,164)
binom.test(987,987+84)
binom.test(1494,1494+145)

Predictions_noBMI_secondadmit <- predict(XGBoostTest_noBMI, as.matrix(SecondAdmit_Clean[,which(colnames(SecondAdmit_Clean) %in% KeepTest$Feature[c(1:25)[-c(2,8,10,14,15,24)]])]), type="response")
confusionMatrix(table("Prediction"=ifelse(Predictions_noBMI_secondadmit>0.05,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_noBMI_secondadmit>0.1,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_noBMI_secondadmit>0.15,1,0), "True"=SecondAdmit_Clean$SBP), positive = "1")

binom.test(157,157+6)
binom.test(1154,1154+57)
binom.test(1841,1841+111)

#VCU / NACSELD Validation
VCU <- read.csv("VCU Validation.csv")
VCU$NeutrophilPercentage <- as.numeric(VCU$NeutrophilPercentage)

VCU$AdmitTemp <- (VCU$AdmitTemp * 9/5) + 32

#Line up column names
XGBoostTest$feature_names
colnames(VCU)

VCU_Features <- VCU[,c(21,12,7,11,3,15,16,14,13,9,4,17,8,10,2,5,6,20,18,19)]

VCU_Features <- as.matrix(apply(VCU_Features, 2, as.numeric))

colnames(VCU_Features) == XGBoostTest$feature_names

Predictions_VCU <- predict(XGBoostTest, VCU_Features, type="response")
Predictions_VCU_noBMI <- predict(XGBoostTest_noBMI, VCU_Features[,-1], type="response")

#Predictions , NACSELD
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU>0.05,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU>0.1,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU>0.15,1,0), "True"=VCU$SBP), positive = "1")

confusionMatrix(table("Prediction"=ifelse(Predictions_VCU_noBMI>0.05,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU_noBMI>0.1,1,0), "True"=VCU$SBP), positive = "1")
confusionMatrix(table("Prediction"=ifelse(Predictions_VCU_noBMI>0.15,1,0), "True"=VCU$SBP), positive = "1")

#CIs for NPV, NACSELD (no BMI)
binom.test(17,17)
binom.test(94,96)
binom.test(150,154)

#CIs for NPV, NACSELD
binom.test(19,19)
binom.test(94,95)
binom.test(147,150)

#DCA
Probabilities_Test <- data.frame("SBP" = Test$SBP, "Model"=Predictions2, "LR"=Predictions_sens2_valid1)
Probabilities_Internal <- data.frame("SBP" = SecondAdmit_Clean$SBP, "Model"=Predictions_second, "LR"=Predictions_sens2_valid2)

A <- net_intervention_avoided(dca(data=Probabilities_Test, SBP ~ Model + LR, thresholds = seq(0, 0.3, by = 0.01))) %>% plot(smooth=T, span=0.5)
A + labs(y="Net Interventions Avoided") 

dca(data=Probabilities_Test, SBP ~ Model + LR, thresholds=seq(0, 0.5, by = 0.01)) %>% plot() + ggplot2::scale_color_manual(labels=c("Treat All", "Treat None", "Model", "LR"), values=c("cyan", "magenta", "yellow", "black")) + theme(legend.position = c(.8, 0.6))

#Covariate Summaries
FeatureUnivariate <- NULL
for (i in 1:20) {
  varName <- XGBoostTest$feature_names[i]
  NoSBP <- paste(round(median(SecondAdmit_Clean[SecondAdmit_Clean$SBP==0, which(colnames(SecondAdmit_Clean) %in% varName)],na.rm=T),5), "(",unname(round(quantile(SecondAdmit_Clean[SecondAdmit_Clean$SBP==0, which(colnames(SecondAdmit_Clean) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(SecondAdmit_Clean[SecondAdmit_Clean$SBP==0, which(colnames(SecondAdmit_Clean) %in% varName)],na.rm=T)[4],5)),")")
  SBP <- paste(round(median(SecondAdmit_Clean[SecondAdmit_Clean$SBP==1, which(colnames(SecondAdmit_Clean) %in% varName)],na.rm=T),5), "(",unname(round(quantile(SecondAdmit_Clean[SecondAdmit_Clean$SBP==1, which(colnames(SecondAdmit_Clean) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(SecondAdmit_Clean[SecondAdmit_Clean$SBP==1, which(colnames(SecondAdmit_Clean) %in% varName)],na.rm=T)[4],5)),")")
  pvalue <- wilcox.test(SecondAdmit_Clean[,which(colnames(SecondAdmit_Clean) %in% varName)] ~ SecondAdmit_Clean$SBP)$p.value
  
  vector <- c("Feature"=varName, 
              "Rank (overall)" = which(importances$Feature %in% varName),"No SBP" = NoSBP, "SBP" = SBP, "p-value" = ifelse(pvalue<0.001, "<0.001",pvalue))
  FeatureUnivariate <- data.frame(rbind(FeatureUnivariate, vector))
}

View(FeatureUnivariate)

CombinedExternalDF <- data.frame(CombinedExternal)
colnames(CombinedExternalDF)[21] <- "SBP"
FeatureUnivariate <- NULL
for (i in 1:20) {
  varName <- XGBoostTest$feature_names[i]
  NoSBP <- paste(round(median(CombinedExternalDF[CombinedExternalDF$SBP==0, which(colnames(CombinedExternalDF) %in% varName)],na.rm=T),5), "(",unname(round(quantile(CombinedExternalDF[CombinedExternalDF$SBP==0, which(colnames(CombinedExternalDF) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(CombinedExternalDF[CombinedExternalDF$SBP==0, which(colnames(CombinedExternalDF) %in% varName)],na.rm=T)[4],5)),")")
  SBP <- paste(round(median(CombinedExternalDF[CombinedExternalDF$SBP==1, which(colnames(CombinedExternalDF) %in% varName)],na.rm=T),5), "(",unname(round(quantile(CombinedExternalDF[CombinedExternalDF$SBP==1, which(colnames(CombinedExternalDF) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(CombinedExternalDF[CombinedExternalDF$SBP==1, which(colnames(CombinedExternalDF) %in% varName)],na.rm=T)[4],5)),")")
  pvalue <- wilcox.test(CombinedExternalDF[,which(colnames(CombinedExternalDF) %in% varName)] ~ CombinedExternalDF$SBP)$p.value
  
  vector <- c("Feature"=varName, 
              "Rank (overall)" = which(importances$Feature %in% varName),"No SBP" = NoSBP, "SBP" = SBP, "p-value" = ifelse(pvalue<0.001, "<0.001",pvalue))
  FeatureUnivariate <- data.frame(rbind(FeatureUnivariate, vector))
}

View(FeatureUnivariate)

#VCU Only
colnames(VCU)[21] <- "SBP"
FeatureUnivariate <- NULL
for (i in 1:20) {
  varName <- XGBoostTest$feature_names[i]
  NoSBP <- paste(round(median(VCU[VCU$SBP==0, which(colnames(VCU) %in% varName)],na.rm=T),5), "(",unname(round(quantile(VCU[VCU$SBP==0, which(colnames(VCU) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(VCU[VCU$SBP==0, which(colnames(VCU) %in% varName)],na.rm=T)[4],5)),")")
  SBP <- paste(round(median(VCU[VCU$SBP==1, which(colnames(VCU) %in% varName)],na.rm=T),5), "(",unname(round(quantile(VCU[VCU$SBP==1, which(colnames(VCU) %in% varName)],na.rm=T)[2],5)), "-", unname(round(quantile(VCU[VCU$SBP==1, which(colnames(VCU) %in% varName)],na.rm=T)[4],5)),")")
  pvalue <- wilcox.test(VCU[,which(colnames(VCU) %in% varName)] ~ VCU$SBP)$p.value
  
  vector <- c("Feature"=varName, 
              "Rank (overall)" = which(importances$Feature %in% varName),"No SBP" = NoSBP, "SBP" = SBP, "p-value" = ifelse(pvalue<0.001, "<0.001",pvalue))
  FeatureUnivariate <- data.frame(rbind(FeatureUnivariate, vector))
}

View(FeatureUnivariate)

#Additional cohort characteristics

prop.table(table(SBP_ML_Clean$Gender))
prop.table(table(SBP_ML_Clean$AlcoholicCirrhosis))
mean(SBP_ML_Clean$Age, na.rm=T)
sd(SBP_ML_Clean$Age, na.rm=T)

prop.table(table(SecondAdmit_Clean$Gender))
prop.table(table(SecondAdmit_Clean$AlcoholicCirrhosis))
mean(SecondAdmit_Clean$Age, na.rm=T)
sd(SecondAdmit_Clean$Age, na.rm=T)

table(SBP_ML_Clean$Gender, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$Gender, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$Gender, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$RaceWhite, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$RaceWhite, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$RaceWhite, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$EthnicityHispanic, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$EthnicityHispanic, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$EthnicityHispanic, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$AlcoholicCirrhosis, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$AlcoholicCirrhosis, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$AlcoholicCirrhosis, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$DiabetesAny2yrs, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$DiabetesAny2yrs, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$DiabetesAny2yrs, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$PPI90days, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$PPI90days, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$PPI90days, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$Betablocker90days, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$Betablocker90days, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$Betablocker90days, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$Rifaximin, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$Rifaximin, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$Rifaximin, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$Lactulose, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$Lactulose, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$Lactulose, SBP_ML_Clean$SBP==1))

table(SBP_ML_Clean$Statin90days, SBP_ML_Clean$SBP==1)
prop.table(table(SBP_ML_Clean$Statin90days, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBP_ML_Clean$Statin90days, SBP_ML_Clean$SBP==1))

SBPPr <- ifelse(SBP_ML_Clean$Fluro==1 | SBP_ML_Clean$Bactrim==1, 1, 0)

table(SBPPr, SBP_ML_Clean$SBP==1)
prop.table(table(SBPPr, SBP_ML_Clean$SBP==1), 2)
chisq.test(table(SBPPr, SBP_ML_Clean$SBP==1))

sd(SBP_ML_Clean[SBP_ML_Clean$SBP==1,"Age"], na.rm=T)
sd(SBP_ML_Clean[SBP_ML_Clean$SBP==0,"Age"], na.rm=T)
t.test(SBP_ML_Clean$Age ~ SBP_ML_Clean$SBP)

#Second Admit

SecondAdmit_Gender <- read.csv("SecondAdmission_Gender.csv")
SecondAdmit_Gender$Gender[SecondAdmit_Gender$Gender=='NULL'] <- NA
SecondAdmit_Gender$Gender <- ifelse(SecondAdmit_Gender$Gender=="M",1,0)
prop.table(table(SecondAdmit_Gender$Gender))

table(SecondAdmit_Gender$Gender, SecondAdmit_Gender$SBP==1)
prop.table(table(SecondAdmit_Gender$Gender, SecondAdmit_Gender$SBP==1), 2)
chisq.test(table(SecondAdmit_Gender$Gender, SecondAdmit_Gender$SBP==1))

table(SecondAdmit_Clean$RaceWhite, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$RaceWhite, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$RaceWhite, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$EthnicityHispanic, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$EthnicityHispanic, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$EthnicityHispanic, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$AlcoholicCirrhosis, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$AlcoholicCirrhosis, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$AlcoholicCirrhosis, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$DiabetesAny2yrs, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$DiabetesAny2yrs, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$DiabetesAny2yrs, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$PPI90days, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$PPI90days, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$PPI90days, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$Betablocker90days, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$Betablocker90days, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$Betablocker90days, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$Rifaximin, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$Rifaximin, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$Rifaximin, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$Lactulose, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$Lactulose, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$Lactulose, SecondAdmit_Clean$SBP==1))

table(SecondAdmit_Clean$Statin90days, SecondAdmit_Clean$SBP==1)
prop.table(table(SecondAdmit_Clean$Statin90days, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SecondAdmit_Clean$Statin90days, SecondAdmit_Clean$SBP==1))

SBPPr <- ifelse(SecondAdmit_Clean$Fluro==1 | SecondAdmit_Clean$Bactrim==1, 1, 0)

table(SBPPr, SecondAdmit_Clean$SBP==1)
prop.table(table(SBPPr, SecondAdmit_Clean$SBP==1), 2)
chisq.test(table(SBPPr, SecondAdmit_Clean$SBP==1))

sd(SecondAdmit_Clean[SecondAdmit_Clean$SBP==1,"Age"], na.rm=T)
sd(SecondAdmit_Clean[SecondAdmit_Clean$SBP==0,"Age"], na.rm=T)
t.test(SecondAdmit_Clean$Age ~ SecondAdmit_Clean$SBP)

