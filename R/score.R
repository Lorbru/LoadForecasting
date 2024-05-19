rmse<-function(y, ychap, digits=0)
{
  return(round(sqrt(mean((y-ychap)^2,na.rm=TRUE)),digits=digits))
}

mape<-function(y,ychap)
{
  return(round(100*mean(abs(y-ychap)/abs(y)),digits=2))
}


rmse.old<-function(residuals, digits=0)
{
  return(round(sqrt(mean((residuals)^2,na.rm=TRUE)),digits=digits))
}


absolute_loss <- function(y, yhat)
{
  mean(abs(y-yhat),na.rm=TRUE)
}

bias <- function(y, yhat)
{
  mean(y-yhat,na.rm=TRUE)
}

R_2 <- function(y, yhat){
  ybar = mean(y)
  SST = sum( (y - ybar)^2 )
  SSR = sum( (yhat - ybar)^2 )
  return(SSR/SST)
}


pinball_loss <- function(y, yhat_quant, quant, output.vect=FALSE)
{
  yhat_quant <- as.matrix(yhat_quant)
  pinball_loss <- 0
  nq <- ncol(yhat_quant)
  loss_q <- array(0, dim=nq)
  
  for (q in 1:nq) {
    loss_q[q] <- mean(((y-yhat_quant[,q]) * (quant[q]-(y<yhat_quant[,q]))), na.rm=T)
    #pinball_loss <- pinball_loss + loss_q /nq
    #print(pinball_loss)
  }
  if(output.vect==FALSE)
  {
    pinball_loss <- mean(loss_q)
  }
  if(output.vect==TRUE)
  {
    pinball_loss <- loss_q
  }
  return(pinball_loss)
  
}


timeCV = function(type, eq, X, y, K, min_train_size=365, quant=0.95, eqRes=NULL, showPL=T){
  
  
  R_2_list = numeric(K)
  rmse_train = numeric(K)
  mape_train = numeric(K)
  rmse_test = numeric(K)
  mape_test = numeric(K)
  pl = numeric(K)
  tsize=numeric(K)
  
  batch = as.integer((length(y)-min_train_size)/K)
  
  # Itération sur les splits
  for (i in 1:K) {
    
    test_index <- (min_train_size + batch*(i-1)):(min_train_size + batch*i)
    train_index <- 0:(min_train_size + (batch*(i-1)))
    tsize[i] = (min_train_size + batch*(i-1))
    
    # Données d'entraînement et de test
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[test_index, ]
    y_test <- y[test_index]
    
    # Performance de regression linéaire
    if (type == "rq"){
      model = rq(eq, data=X_train, tau=quant)
      yhat_train = predict(model, newdata=X_train)
      yhat_test = predict(model, newdata=X_test)
    } else if (type == "lm") {
      model = lm(eq, data=X_train)
      yhat_train = predict(model, newdata=X_train)
      yhat_test = predict(model, newdata=X_test)
      
    } else if (type == "gam") {
      model = gam(eq, data=X_train)
      yhat_train = predict(model, newdata=X_train)
      yhat_test = predict(model, newdata=X_test)
    } else if (type == "rf") {
      model = ranger(eq, data=X_train, importance='permutation')
      yhat_train = predict(model, data=X_train)$predictions
      yhat_test = predict(model, data=X_test)$predictions
    } else if (type == "gamrf") {
      
      gam_model = gam(eq, data=X_train)
      residuals = residuals(gam_model)
      df_res = cbind(X_train, residuals=residuals)
      rf_model = ranger(eqRes, data=df_res, importance="permutation")
      
      yhat_train = predict(gam_model, newdata=X_train) + predict(rf_model, data=X_train)$predictions
      yhat_test = predict(gam_model, newdata=X_test) + predict(rf_model, data=X_test)$predictions
 
    }
    
    #
    R_2_list[i] = R_2(y_train, yhat_train)
    rmse_train[i] = rmse(y_train, yhat_train)
    rmse_test[i] = rmse(y_test, yhat_test)
    mape_train[i] = mape(y_train, yhat_train)
    mape_test[i] = mape(y_test, yhat_test)
    
    pl[i] = pinball_loss(y_test, yhat_test, quant)
    
  }
  if (showPL){
    return(data.frame("tsize" = tsize, "R_2"=R_2_list, "RMSE_train"=rmse_train, "RMSE_test"=rmse_test, "MAPE_train"=mape_train, "MAPE_test"=mape_test, "PL"=pl)[1:K,])
  } else {
    return(data.frame("tsize" = tsize, "R_2"=R_2_list, "RMSE_train"=rmse_train, "RMSE_test"=rmse_test, "MAPE_train"=mape_train, "MAPE_test"=mape_test)[1:K,])
  }
  
  
}


