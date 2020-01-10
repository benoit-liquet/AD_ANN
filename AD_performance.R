# ADPerformance: calculate and return performance metrics
# output of regression/classification method:
# Binary Matrix of known anomalies, each column corresponds to a variable
ADPerformance2 <- function(Output, Truth, print_out = T){
  # pm <- setNames(data.frame(matrix(ncol = 12, nrow = 0)),
  #                c("TN", "FN", "FP", "TP", "Accuracy", "Error_Rate", "Sensitivity",
  #                  "Specificity", "Precision", "Recall", "F_Measure", "Optimised_Precision"))
  pm <- setNames(data.frame(matrix(ncol = 12, nrow = 0)),
                 c("TN", "FN", "FP", "TP", "Accuracy", "Error_Rate", "Geometric mean", "NPV", "PPV", "bal_Accuracy", "f1", "MCC"))
  j <- 1
  
  Output <- Output[!is.na(Output),]
  Truth <- Truth[!is.na(Output),]
  
  for (i in colnames(Output)[2:ncol(Output)]) { # Timestamp is the first column in both dataframes
    # Get confusion matrix
    
    TN   <- sum(Output[,i] == 0 & Truth[,i] == 0, na.rm = T)
    
    FN   <- sum(Output[,i] == 0 & Truth[,i] == 1, na.rm = T)
    
    FP   <- sum(Output[,i] == 1 & Truth[,i] == 0, na.rm = T)
    
    TP   <- sum(Output[,i] == 1 & Truth[,i] == 1, na.rm = T)
    
    acc  <- (TP + TN) / (TP + FP + TN + FN)
    
    err  <- (FP + FN) / (TP + FP + TN + FN)
    
    sn   <- TP / (TP + FN) #Sensitivity
    
    sp   <- TN / (TN + FP) #Specificity
    
    p    <- TP / (TP + FP) #Precission
    
    r    <- TP / (TP + TN) #Recall
    
    #FM   <- (2*p*r) / (p + r)
    
    GM   <- sqrt(TP * TN)
    
    PPV <- TP/(TP + FP) #Positive Predicted Value
    if (is.na(PPV)) PPV <- 0  
    
    NPV <- TN/(TN + FN) #Negative Predicted Value
    if (is.na(NPV)) NPV <- 0  
    
    b_acc <- (sn + sp) / 2 #If P >> N, b_acc is better
    
    f1 <- 2 * p * sn / (p + sn) # If N >> P, f1 is a better. But it does not include TN in formula  
    if (is.na(f1)) f1 <- 0 
    
    N <- TN + TP + FN + FP
    
    MCC <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (FP + TN) * (TN + FN)) #good performance for imbalance data
    if (is.na(MCC)) MCC <- 0
    
    # Nn   <- (FP + TN) / (TP + FP + TN + FN)
    # Np   <- (TP + FN) / (TP + FP + TN + FN)
    # P    <- (sp*Nn + sn*Np)
    # RI   <- (abs(sp - sn)/(sp+sn))
    # OP   <- P - RI
    # 
    
    # OP   <- acc - (abs(sp - sn)/(sp+sn))
    # 
    # sss = FN+TP/(FN+TP+TN+FP); 
    # ssa = 1 - sss; 
    # M = sss*sn + ssa*sp; 
    # OPP = M - (abs(sp - sn)/(sp+sn))
    
    pm[j,] <- c(TN, FN, FP, TP, acc, err, sn, sp, GM, NPV, PPV, b_acc, f1, MCC)
    j <- j + 1
    
  }
  rownames(pm) <- colnames(Output)[2:ncol(Output)]
  if(print_out)
  {print(round(pm,4))}
  
  return(pm)
  
}
