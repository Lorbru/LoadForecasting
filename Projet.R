# =========================================
#
#     Projet - Net demand forecasting
#
# =========================================


# ========================
# 0. Librairies
# ========================

rm(list=objects())
setwd("~/Projects/NetDemandForecasting")
source("R/score.R")
library(lubridate)
library(mgcv)
library(yarrr)
library(magrittr)
library(caret)
library(forecast)
library(tidyverse)
library("quantreg")

################################################################################

# ========================
# 1. Données
# ========================

# train set et test set
train = read_csv("Data/train.csv")
test = read_csv("Data/test.csv")

# feature engineering
train$WeekDays = as.factor(train$WeekDays)
test$WeekDays = as.factor(test$WeekDays)

train$Time = as.numeric(train$Date)
test$Time = as.numeric(test$Date)

train$WeekDays2 <- weekdays(train$Date)
train$WeekDays3 <- forcats::fct_recode(train$WeekDays2, 'WorkDay'='mardi' ,'WorkDay'='mercredi', 'WorkDay' = 'jeudi')
test$WeekDays2 <- weekdays(test$Date)
test$WeekDays3 <- forcats::fct_recode(test$WeekDays2, 'WorkDay'='mardi' ,'WorkDay'='mercredi', 'WorkDay' = 'jeudi')

train$WeekDays2 <- as.factor(train$WeekDays2)
train$WeekDays3 <- as.factor(train$WeekDays3)
test$WeekDays2 <- as.factor(test$WeekDays2)
test$WeekDays3 <- as.factor(test$WeekDays3)

train$Month <- months(train$Date)
test$Month <- months(test$Date)

train$Season <- forcats::fct_recode(train$Month,
                                    'hiver'='décembre', 'hiver'='janvier', 'hiver'='février',
                                    'printemps'='mars', 'printemps'='avril', 'printemps'='mai',
                                    'ete'='juin', 'ete'='juillet', 'ete'='août',
                                    'automne'='septembre','automne'='octobre', 'automne'='novembre')
test$Season <- forcats::fct_recode(test$Month,
                                    'hiver'='décembre', 'hiver'='janvier', 'hiver'='février',
                                    'printemps'='mars', 'printemps'='avril', 'printemps'='mai',
                                    'ete'='juin', 'ete'='juillet', 'ete'='août',
                                    'automne'='septembre','automne'='octobre', 'automne'='novembre')


train$Winter <- forcats::fct_recode(train$Month,
                                    'winter'='décembre', 'winter'='janvier', 'noWinter'='février',
                                    'noWinter'='mars', 'noWinter'='avril', 'noWinter'='mai',
                                    'noWinter'='juin', 'noWinter'='juillet', 'noWinter'='août',
                                    'noWinter'='septembre','noWinter'='octobre', 'noWinter'='novembre')

test$Winter <- forcats::fct_recode(test$Month,
                                    'winter'='décembre', 'winter'='janvier', 'noWinter'='février',
                                    'noWinter'='mars', 'noWinter'='avril', 'noWinter'='mai',
                                    'noWinter'='juin', 'noWinter'='juillet', 'noWinter'='août',
                                    'noWinter'='septembre','noWinter'='octobre', 'noWinter'='novembre')

train$Temp_trunc1 <- pmax(train$Temp-285,0)
train$Temp_trunc2 <- pmax(train$Temp-290,0)

test$Temp_trunc1 <- pmax(test$Temp-285,0)
test$Temp_trunc2 <- pmax(test$Temp-290,0)

inCovid = function(x){
  return(ifelse((x >= '2020-03-17' && x <= '2020-05-11') | 
           (x >= '2020-10-30' && x <= '2020-12-15') |
           (x >= '2021-04-03' && x <= '2021-05-03'), 1, 0))
}

train$Covid = sapply(train$Date, inCovid)
test$Covid = sapply(test$Date, inCovid)

train = train[which(train$Date >= '2018-01-01'),]


# variables réponses (jeu d'entrainement)
# Net_demand = load - solar_power - wind_power
ytrain_nd = train$Net_demand
ytrain_sp = train$Solar_power
ytrain_wp = train$Wind_power
ytrain_ld = train$Load

train$Temp

################################################################################

# ========================
# 2. Visualisation
# ========================

par(mfrow=c(1, 1))
plot(Net_demand~Date,  data=train, type='l', col='red')

plot(Net_demand~Date, data=train, col=Covid)
lines(Net_demand~Date, data=train)

################################################################################

# ========================
# 3. Modèles linéaires
# ========================

# Modèles de regression linéaire :
# ================================

# Premier essai en fonction des variables de décalage temporel
eq1 = Net_demand ~ Time + Load.1 + Load.7 + Wind_power.1 + Wind_power.7 + Net_demand.1 + Net_demand.7 + Solar_power.1 + Solar_power.7
model = lm(eq1, data=train)
summary(model)
timeCV("lm", eq1, train, train$Net_demand, 8)

# le summary révèle que les variables avec sept jours de décalage sont peu significatives pour la produciton éolienne et solaire 
# (pas de saisonalité particulière sur les semaines pour des phénomènes naturels)
# Modèle en fort sous apprentissage (variabilité peu expliquée, erreur train/test de CV élevée)



# Ajouts en analyse ascendente de variables plus significatives
eq2 = Net_demand ~ Time + Net_demand.1 + Load.1 + Wind_power.1 + WeekDays3 + Covid + WeekDays3 + Covid +
  Nebulosity_weighted +  Wind_weighted + Temp
model = lm(eq2, data=train)
summary(model)
timeCV("lm", eq2, train, train$Net_demand, 8)

# Significativité du modèle par rapport au précédent par test RV
anova(lm(eq2, data=train), lm(eq1, data=train)) 





# Modèle de regression final :
eqLin = Net_demand ~ Time + Load.1 + Wind_power.1 + WeekDays3 + Covid + 
  Nebulosity_weighted +  Wind_weighted + Temp + Temp_s95 + Season + Summer_break + Christmas_break + DLS + BH_Holiday + 
  Holiday + BH_before + Wind


model = lm(eqLin, data=train)
summary(model)
timeCV("lm", eqLin, train, train$Net_demand, 5)


anova(lm(eqLin, data=train), lm(eq2, data=train))




# Modèles de regression quantile :
# ================================ 

rqTimeCV(eq7, train, train$Net_demand, 2, 0.57)

rqmod = rq(eq7, tau=0.57, data=train)
forecast = predict(rqmod, newdata=test)
lines(forecast~test$Date, col='green')

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- forecast
write.table(submit, file="Data/submission_rq_cv2.csv", quote=F, sep=",", dec='.',row.names = F)

################################################################################

# ========================
# 4. Modèles additifs
# ========================

k1 = 5


# equation gam issue de la régression linéaire (lissage de la régression linéaire)
eqGam1 = Net_demand ~ s(Time, k=3, bs='cr') + s(Load.1, k=k1) + s(Wind_power.1, k=k1) + Covid + WeekDays3 + Covid +
  s(Nebulosity_weighted, k=k1, bs='cr') + s(Wind_weighted, k=k1, bs='cr') + s(Temp, k=k1, bs='cr') + 
  s(Temp_s95, k=3) + Season + Summer_break + Christmas_break + DLS + BH_Holiday + 
  Holiday + BH_before


modGam = gam(eqGam1, data=train)
summary(modGam)
timeCV("gam", eqGam1, train, train$Net_demand, 5)


# nouvelle analyse ascendante pour es gam :
eqGam2 = Net_demand ~ s(Time, k=3, bs='cr') + s(Load.1, k=k1) + Covid + WeekDays3 + Covid +
  s(Nebulosity_weighted, k=k1, bs='cr') + s(Wind_weighted, k=k1, bs='cr') + s(Temp, k=k1, bs='cr') + 
  s(Temp_s95, k=3) + Summer_break + Christmas_break + BH_Holiday + 
  Holiday + BH_before + 
  s(Temp_s99_min, k=3) + s(Temp_s99_max, k=3) + te(Temp_s95_min, Temp_s95_max) +
  te(Wind, Solar_power.1) + te(Temp, Solar_power.1) +
  s(toy, k=10, bs='cc') +
  Holiday +
  s(Net_demand.1, k=3, bs='cr') + 
  te(Load.1, Load.7) + 
  te(Wind_power.1, Wind_power.7)

source("R/score.R")


modGam = gam(eqGam2, data=train)
summary(modGam)


timeCV("gam", eqGam2, train, train$Net_demand, 8)

anova(gam(eqGam1, data=train), gam(eqGam2, data=train))

eq8 = Net_demand ~ s(Time,k=3, bs='cr') + te(Load.1, Load.7) + s(Solar_power.7, bs='cr') + s(Wind_power.1, bs='cr') + s(Net_demand.1) + WeekDays3 + Covid +
  s(Nebulosity_weighted) +  s(Wind_weighted) + te(Temp, Temp_s95) + Season + Summer_break + Christmas_break + DLS + BH_Holiday

eq9 = Net_demand ~ s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.7 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Season + Summer_break + Christmas_break + DLS + BH_Holiday 

eq10 = Net_demand ~ s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.7 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Season + Summer_break + Christmas_break + DLS + BH_Holiday + s(toy, k=30, bs='cc') +
  + Holiday + Holiday_zone_a + Holiday_zone_b + Holiday_zone_c + BH + BH_before + BH_after

eq10 = Net_demand ~ s(as.numeric(Date), k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.1 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Season + Summer_break + Christmas_break + DLS + BH_Holiday + s(toy, k=20, bs='cc') +
  + Holiday + Holiday_zone_a + Holiday_zone_b + Holiday_zone_c +
  BH + BH_before + BH_after

plot(Net_demand~Wind_weighted, data=train)
lines(predict(gam(Net_demand~s(Wind_weighted, k=3), data=train), newdata=train)~Wind_weighted, data=train)

plot(Net_demand~Temp, data=train)

summary(gam(eq10, data=train))

timeCV("gam", eq10, train, train$Net_demand, 5)
eq10 = Net_demand ~ s(as.numeric(Date), k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.1 + Wind_power.1 + Net_demand.1 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Summer_break + Christmas_break + DLS + BH_Holiday + s(toy, k=20, bs='cc') +
  + Holiday + Holiday_zone_b +  BH + BH_before + BH_after

gmod = gam(eq10, data=Data0)
summary(gmod)

forecast = predict(gmod, newdata=Data1)
plot(forecast~test$Date, col='darkgreen') # à essayer 
lines(forecast~test$Date, col='red')
submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- forecast
write.table(submit, file="Data/submission_gam2.csv", quote=F, sep=",", dec='.',row.names = F)



################################################################################

# ========================
# 5. Modèles ensemblistes
# ========================

eqLin = Net_demand ~ Time + Load.1 + Wind_power.1 + WeekDays3 + Covid + 
  Nebulosity_weighted +  Wind_weighted + Temp + Temp_s95 + Season + Summer_break + Christmas_break + DLS + BH_Holiday + 
  Holiday + BH_before + Wind

# Random Forest
model = ranger(eqLin, data=train, importance='permutation')
timeCV("rf", eqLin, train, train$Net_demand, 5)

source("R/score.R")

model$variable.importance
# Créer un dataframe pour l'importance des variables
var_importance <- data.frame(Variable = row.names(model$variable.importance),
                             Importance = model$variable.importance)


# Trier par importance
var_importance <- var_importance[order(-var_importance$Importance),]

# Créer un graphique d'importance des variables
barplot(height = model$variable.importance, names.arg = names(model$variable.importance), las = 2, horiz = TRUE, col = "steelblue", main = "Importance des variables avec Random Forest", xlab = "Importance")



# Random forest + gam
library(ranger)


# eq11
eq11 = Net_demand~s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') +
  Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + s(Temp, k=10, bs="cr") + s(Temp_s95, k=10, bs="cr") + 
  DLS + BH_Holiday + BH_before + BH_after + s(toy, k=30, bs='cc') + Season

eq11res = residuals~Time + Load.1 + Load.7 + Net_demand.1 + Net_demand.7 + Solar_power.1 +
  Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + Temp + Temp_s95 +
  DLS + BH_Holiday + BH_before + BH_after + toy + Season

# eq12
eq12 = Net_demand~s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') +
  Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + s(Temp, k=10, bs="cr") + 
  DLS + BH_Holiday + BH_before + BH_after + Summer_break

eq12res = residuals~Time + Load.1 + Load.7 + Net_demand.1 + Net_demand.7 + Solar_power.1 +
  Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + Temp +
  DLS + BH_Holiday + BH_before + BH_after + Summer_break

# eq13
eq13 <- Net_demand~s(as.numeric(Date),k=6, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') + 
  te(Solar_power.1, Wind_power.1, bs='cr') + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=6, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=35, bs='cc') + Temp_trunc2 + Season + s(Wind, k=6)# (final : 123)

eq13res = residuals~ Time + Load.1 + Load.7 + Net_demand.1 + Net_demand.7 + 
  Solar_power.1 + Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + Temp + Temp_s95 + 
  BH_Holiday + BH_before + toy + Temp_trunc2 + Season + Wind# (final : 123)

library(ranger)
timeCV("gamrf", eq13, train, train$Net_demand, 8, eqRes = eq13res)
model = gam(eq13, data=train)

summary(model)

# eq14
eq14 <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') + 
  te(Solar_power.1, Wind_power.1, bs='cr') + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + te(Temp_s95_max, Temp_s95_min) + te(Temp_s99_max, Temp_s99_min)

eq14res <- residuals~Time + Load.1 + Load.7 + Temp + Temp_s95 + Temp_s99 + Temp_s95_min + Temp_s95_max + Temp_s99_min +
  Temp_s99_max + Wind_weighted + Nebulosity_weighted + toy + WeekDays + BH_before + BH + BH_after + Year + Month

# eq15
eq15 <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') + 
  te(Solar_power.1, Wind_power.1, bs='cr') + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + te(Temp_s95_max, Temp_s95_min) + te(Temp_s99_max, Temp_s99_min) + Month + Season + Temp_trunc2

eq15res <- residuals~Time + Load.1 + Load.7 + Temp + Temp_s95 + Temp_s99 + Temp_s95_min + Temp_s95_max + Temp_s99_min +
  Temp_s99_max + Wind_weighted + Nebulosity_weighted + toy + WeekDays + BH_before + BH + BH_after




timeCV("gam", eq15, train, train$Net_demand, 8)

graphics.off()
plot()

gam_model = gam(eq15, data=train)
summary(gam_model)
residuals = residuals(gam_model)
df_res = cbind(train, residuals=residuals)
rf_model = ranger(eq15res, data=df_res, importance="permutation")
rf_model$variable.importance

yhat_train = predict(gam_model, newdata=train) + predict(rf_model, data=train)$predictions
yhat_test = predict(gam_model, newdata=test) + predict(rf_model, data=test)$predictions

lines(yhat_test~Date, data=test, col='red')

gam_mod = gam(eq11, data=train)
res = residuals(gam_mod)
df_res = cbind(train, residuals=res)
rf_mod = ranger(eq11res, data=df_res, importance="permutation")
forecast = predict(gam_mod, newdata=train) + predict(rf_mod, newdata=train)$predictions



submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- yhat_test
write.table(submit, file="Data/submission_gamrf_+month+season+tt.csv", quote=F, sep=",", dec='.',row.names = F)


#### Recherche affinée d'un gam performant


fitModConc = Net_demand ~ s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.7 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Season + Summer_break + Christmas_break + DLS + BH_Holiday + s(toy, k=5, bs='cc') +
  + Holiday + Holiday_zone_a + Holiday_zone_b + Holiday_zone_c + BH + BH_before + BH_after

bestMod = Net_demand ~ s(Time, k=3, bs='cr') + 
  s(Net_demand.1, k=3, bs='cr') + s(Load.1, k=3, bs='cr') + s(Load.7, k=3, bs='cr') +
  te(Solar_power.1, Temp) + 
  s(Temp, k=5, bs='cr') + s(Wind_weighted, k=5, bs='cr') +
  te(Temp_s95_min, Temp_s95_max) +
  s(Temp_s99_min, k=5, bs='cr') + s(Temp_s99_max, k=5, bs='cr') +
  WeekDays3 + Covid +
  Summer_break + Christmas_break +
  BH_before + BH_Holiday + Holiday +
  Holiday_zone_b + s(toy, k=5, bs='cc') + 
  Nebulosity_weighted + Month

resMod = residuals ~ Time + 
  Net_demand.1 + Load.1 + Load.7 +
  Solar_power.1 + Temp
  

model = gam(bestMod, data=train)
res = residuals(model)
rf = ranger(resMod, data=cbind(train, residuals))
forecast = predict(model, newdata=test) + predict(rf, data=test)$predictions

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- forecast
write.table(submit, file="Data/submission_gamrf.csv", quote=F, sep=",", dec='.',row.names = F)


model = ranger(Net_demand~Net_demand.1, data=train)
q95pred = predict(model, newdata = train, quantiles=0.95, type = "quantiles")$predictions
q95pred

resEq = residuals~Load.1 + Time + 
  Net_demand.1 + Load.1 + Load.7 +
  Solar_power.1 + Temp + 
  Wind_weighted +
  WeekDays3 + Covid +
  Summer_break + Christmas_break +
  BH_before + BH_Holiday + Holiday +
  Holiday_zone_b + toy + 
  Nebulosity_weighted + Month


model = gam(bestMod, data=train)
summary(model)
residuals = residuals(model)
dfres = cbind(train, residuals)
resrf = ranger(resEq, data=dfres, importance="permutation")

plot(sort(resrf$variable.importance))
sort(resrf$variable.importance)

timeCV("gam", bestMod, train, train$Net_demand, 5)
timeCV("gamrf", bestMod, train, train$Net_demand, 5, eqRes=resEq)

timeCV("gamrf", eq13, train, train$Net_demand, 5, eqRes=eq13res)


y.forecast = predict(model, newdata=test)

plot(y.forecast~test$Date)


submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- y.forecast
write.table(submit, file="Data/submission_gam.csv", quote=F, sep=",", dec='.',row.names = F)


modelConc = gam(eq13, data=train)
summary(model)
residuals = residuals(modelConc)
dfres = cbind(train, residuals)
resrfConc = ranger(eq13res, data=dfres, importance="permutation")

yconc = predict(modelConc, newdata=test) + predict(resrfConc, data=test)$predictions

lines(yconc~test$Date, col='red')


################################################################################


# 135 rfgam :
equation <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') +
  Solar_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') +
  Summer_break + DLS + BH_Holiday + BH_before + s(toy,k=30, bs='cc')

# 123 rfgam :
equation <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') +
  s(Solar_power.1, k=3, bs='cr') + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') +
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + Temp_trunc2 + Season

eqRes = Net_demand ~Time + toy + Temp + Temp_s99 + Temp_s95_max +Temp_s99_max + Summer_break +
  Christmas_break + Solar_power.1 + Wind_power.1 + Temp_s95_min +Temp_s99_min + DLS  +
  Covid  + Season

eq = equation

timeCV("gamrf", eq, train, train$Net_demand, 8, eqRes = eqRes)

train_set = train[which(train$Year <= 2021),]
valid_set = train[which(train$Year > 2021), ]

library(qgam)


model = gam(equation, data=train_set)
summary(model)
valid_set
qpredVal = predict(model, newdata=valid_set)
qpredTrain = predict(model, newdata=train_set)
pinball_loss(train_set$Net_demand, qpredTrain, 0.95)
pinball_loss(valid_set$Net_demand, qpredVal, 0.95)

forecast = predict(model, newdata=test)
lines(forecast~test$Date, type='l', col='blue')

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- forecast
write.table(submit, file="Data/submission_qgam94.csv", quote=F, sep=",", dec='.',row.names = F)

equation <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') +
  s(Wind_power.1, bs='cr', k=3) + te(Wind_power.1, Solar_power.1) +
  WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') +
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + Temp_trunc2

eqRes = res ~Time + toy + Temp + Temp_s99 + Temp_s95_max +Temp_s99_max + Summer_break +
  Christmas_break + Solar_power.1 + Wind_power.1 + Temp_s95_min +Temp_s99_min + DLS +
  Covid + Season

eq15 <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') + 
  te(Solar_power.1, Wind_power.1, bs='cr') + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + te(Temp_s95_max, Temp_s95_min) + te(Temp_s99_max, Temp_s99_min) +  Temp_trunc2 + Winter

eq15res <- residuals~Time + Load.1 + Load.7 + Temp + Temp_s95 + Temp_s99 + Temp_s95_min + Temp_s95_max + Temp_s99_min +
  Temp_s99_max + Wind_weighted + Nebulosity_weighted + toy + WeekDays + BH_before + BH + BH_after

timeCV("gam", eq15, train, train$Net_demand, 5, eqRes = eq15res)

bestMod = gam(eq15, data=train)
summary(bestMod)
forecastbest = predict(bestMod, newdata=test)
lines(forecastbest~test$Date, col='red')


model = gam(eq, data=train)
res = residuals(model)
dfres = cbind(train, res)
rf = ranger(eqRes, data=dfres)
foretest = predict(model, newdata=test) + predict(rf, data=test)$predictions
plot(foretest~test$Date, col='black', type='l')
