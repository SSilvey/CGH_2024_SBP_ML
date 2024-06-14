
#Get hyperparameters
#Make sure "Train" is defined (see other script)

Try <- expand.grid("eta"=c(0.01,0.05,0.1,0.3),
                   "depth"=c(2,4,6,8),
                   "subsample"=c(0.25,0.5,0.75,1),
                   "gamma"=c(0,1,5,10)
                    )

set.seed(2023)
#Run the grid search
ComboAUC <- NULL
for (i in 1:nrow(Try)) {
  
  params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 eta=Try$eta[i],
                 gamma=Try$gamma[i],
                 max_depth=Try$depth[i],
                 min_child_weight=1,
                 subsample=Try$subsample[i]
                 )
  
  AUCs <- NULL
  for (repeats in 1:5) {
  RandomShuffleData <- Train[sample(nrow(Train)),]
  
  folds <- cut(seq(1,nrow(RandomShuffleData)),breaks=5,labels=F)
  
  for (k in 1:5) {
    testIndices <- which(folds==k,arr.ind=T)
    testData <- RandomShuffleData[testIndices,]
    trainData <- RandomShuffleData[-testIndices,]
    
    XGB <- xgboost(params=params,
                   data=as.matrix(trainData[,-ncol(trainData)]),
                   missing=NA,
                   label=as.numeric(trainData$SBP)-1,
                   nrounds=5,
                   eval_metric="auc", verbose=F
    )
    
    Test_AUC <- roc(response=testData$SBP, 
                    predictor=predict(XGB, as.matrix(testData[,1:(ncol(trainData)-1)]), type="response"), percent=T,
                    quiet=T)
    
    AUCs <- append(AUCs, as.numeric(Test_AUC$auc))
  }
  }

 ComboAUC <- append(ComboAUC, mean(AUCs))
  
}

Try$AUC_CV <- ComboAUC

which.max(ComboAUC)

Try[which.max(Try$AUC_CV),] #Returns the hyperparameter combination
#that maximzed the cross-validated AUC.

#Get nrounds

nrounds_list <- c(5,10,50,100,250,500,1000,1500,2000,2500)

DefaultParams <- list(booster = "gbtree",
                      objective = "binary:logistic",
                      eta=0.05,
                      gamma=5,
                      max_depth=6,
                      min_child_weight=1,
                      subsample=0.75)

set.seed(2023)
NROUND_AUC <- NULL
for (i in 1:length(nrounds_list)) {
  
  AUCs <- NULL
  for (repeats in 1:5) {
  
  RandomShuffleData <- Train[sample(nrow(Train)),]
  
  folds <- cut(seq(1,nrow(RandomShuffleData)),breaks=5,labels=F)
  
  for (k in 1:5) {
    testIndices <- which(folds==k,arr.ind=T)
    testData <- RandomShuffleData[testIndices,]
    trainData <- RandomShuffleData[-testIndices,]
    
    XGB <- xgboost(params=DefaultParams,
                   data=as.matrix(trainData[,-ncol(trainData)]),
                   missing=NA,
                   label=as.numeric(trainData$SBP)-1,
                   nrounds=nrounds_list[i],
                   eval_metric="auc",verbose=F
    )
    
    Test_AUC <- roc(response=testData$SBP, 
                    predictor=predict(XGB, as.matrix(testData[,1:(ncol(trainData)-1)]), type="response"), percent=T,
                    quiet=T)
    
    AUCs <- append(AUCs, as.numeric(Test_AUC$auc))
  }
  }
  NROUND_AUC <- append(NROUND_AUC, mean(AUCs))
}

plot(x=nrounds_list, y=NROUND_AUC, xlab="Ensemble Size", ylab="AUC (held-out partition)")
lines(x=nrounds_list, y=NROUND_AUC)
which.max(NROUND_AUC)


