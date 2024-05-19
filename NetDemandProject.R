################################################################################
#                                                                              #
#  PROJET DE MODELISATION PREDICTIVE : CONSOMMATION NETTE EN PERIODE DE        #
#  SOBRIETE ENERGETIQUE                                                        #
#                                                                              #
################################################################################

# librairies
rm(list=objects())
library(lubridate)
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
library(ranger)
library(gbm)
library(qgam)
library("quantreg")
source("R/score.R")

# Jeu entrainement et test
Data0 = read_csv("Data/train3.csv")
Data1 = read_csv("Data/test3.csv")

sel_a = which(Data0$Year <= 2021)
sel_b = which(Data1$Year > 2021)

################################################################################
#                       VISUALISATION  DES DONNEES                             #
################################################################################

# Covid effects
plot(Data0$Net_demand~Data0$Date, col=Data0$Covid, main="Covid effects")
lines(Data0$Net_demand~Data0$Date, col='red')

# Temperature plot anti-correlation
plot(Data0$Date, Data0$Net_demand, type='l')
par(new=T)
plot(Data0$Date, Data0$Temp, type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4,col='red', col.axis='red')
mtext(side = 4, line = 3, 'Temperature', col='red')
legend("top",c("Net_demand","Temperature"),col=c("black","red"),lty=1,ncol=1,bty="n")

# Temp_trunc
plot(Data0$Temp_trunc2~Data0$Date, type='l', main="Temp trunc")

# Price 
plot(Data0$Price~Data0$Date, type='l', main="Price")

# Inflation
plot(Data0$Inflation~Data0$Inflation, type='l', main='Inflation')

# Correlation plot
plot(Data0$Net_demand~Data0$Temp, main="Net_demand ~ Temp")

plot(Data0$Net_demand~Data0$Net_demand.1, main="Net_demand ~ Net_demand.1")

plot(Data0$Net_demand~Data0$Net_demand.1, main="Net_demand ~ Net_demand.7")

plot(Data0$Net_demand~Data0$Load.1, main="Net_demand ~ Load.1")

plot(Data0$Net_demand~Data0$Load.7, main="Net_demand ~ Load.7")

plot(Data0$Net_demand~Data0$Solar_power.1, main="Net_demand ~ Solar_power.1")

plot(Data0$Net_demand~Data0$Solar_power.7, main="Net_demand ~ Solar_power.7")

plot(Data0$Net_demand~Data0$Wind_power.1, main="Net_demand ~ Wind_power.1")

plot(Data0$Net_demand~Data0$Wind_power.7, main="Net_demand ~ Wind_power.7")

plot(Data0$Net_demand~Data0$Nebulosity, main="Net_demand ~ Nebulosity")

################################################################################
#                            MODELES LINEAIRES                                 #
################################################################################


# Modèles linéaires simples
# =========================

# Equations sur les lag features
eq1 = Net_demand ~ as.numeric(Date) + Load.1 + Load.7 + Wind_power.1 + Wind_power.7 +
  Net_demand.1 + Net_demand.7 + Solar_power.1 + Solar_power.7

timeCV("lm", eq1, Data0, Data0$Net_demand, 4)
pinball_loss(predict(lm(eq1, Data0[sel_a,]), newdata=Data0[sel_b,]), Data0[sel_b,]$Net_demand, quant=0.95)

# Effets des conditions météorologiques 
eq2 = Net_demand ~ as.numeric(Date) + Net_demand.1 + WeekDays3 + Covid + WeekDays3 +
  Nebulosity_weighted +  Wind_weighted + Temp

lm2 = lm(eq2, Data0[sel_a,])
summary(lm2)
timeCV("lm", eq2, Data0, Data0$Net_demand, 4)
pinball_loss(predict(lm2, newdata=Data0[sel_b,]), Data0[sel_b,]$Net_demand, quant=0.95)

# Effets des jours feriés 
eq3 = Net_demand ~ as.numeric(Date) + Load.1 + Wind_power.1 +
  Nebulosity_weighted +  Wind_weighted + Temp + Temp_s95 +
  Summer_break + Christmas_break + DLS + BH_Holiday + Holiday + 
  BH_before + Wind

lm3 = lm(eq3, Data0[sel_a,])
summary(lm3)
timeCV("lm", eq3, Data0, Data0$Net_demand, 4)
pinball_loss(predict(lm3, newdata=Data0[sel_b,]), Data0[sel_b,]$Net_demand, quant=0.95)

# Plot des résidus
regMod = lm(eq3, Data0)
res = residuals(regMod)
plot(res~Data0$Date, main="Residuals on eq3") # saisonalité/variabilité hiver

qqnorm(res, main="Normal QQplot on residuals") # hypothèse gaussienne
qqline(res)


################################################################################
#                            MODELES ADDITIFS                                  #
################################################################################

# Lissage
gamTemp = gam(Net_demand ~ s(Temp, k=10, bs='cr'), data=Data0)
plot(Data0$Net_demand ~ Data0$Temp, main='Net_demand ~ s(Temp, k=10)')
points(predict(gamTemp, data=Data0)~Data0$Temp, col='red')

gamToy = gam(Net_demand ~ s(toy, k=20, bs='cc'), data=Data0)
plot(Data0$Net_demand ~ Data0$toy, main='Net_demand ~ s(toy, k=20)')
points(predict(gamToy, data=Data0)~Data0$toy, col='red')

gamTemp_s99 = gam(Net_demand ~ s(Temp_s99, k=10, bs='cr'), data=Data0)
plot(Data0$Net_demand ~ Data0$Temp_s99, main='Net_demand ~ s(Temp_s99, k=10)')
points(predict(gamTemp_s99, data=Data0)~Data0$Temp_s99, col='red')

# tendance sur l'évolution globale depuis 2018
gamDate = gam(Net_demand ~ s(as.numeric(Date), k=3, bs='cr'), data=Data0)
plot(Data0$Net_demand ~ Data0$Date, main='Net_demand ~ s(Date, k=3)')
points(predict(gamDate, data=Data0)~Data0$Date, col='red')

# Modele gam 1
eqGam1 = Net_demand~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=20, bs='cc') + 
  s(Temp,k=10, bs='cr') + te(Net_demand.1,Net_demand.7, k=10, bs='cr') + 
  s(Temp_s99,k=10, bs='cr') + WeekDays2 + BH +
  te(Nebulosity_weighted,Wind_weighted,k=10, bs='cr') + Covid

# résultats peu pertinents avec min_train_size = 365, augmenation et subdivision
timeCV("gam", eqGam1, Data0, Data0$Net_demand, 4, min_train_size=633)

# Modele gam 2 + BH_before
eqGam2 = Net_demand~s(as.numeric(Date),k=3, bs='cr') + s(toy, k=20, bs='cc') +
  s(Temp,k=10, bs='cr') + te(Net_demand.1,Net_demand.7, k=10, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + WeekDays2 + BH + 
  te(Nebulosity_weighted,Wind_weighted,k=10, bs='cr') + Covid + BH_before

timeCV("gam", eqGam2, Data0, Data0$Net_demand, 4, min_train_size=633)

# gam3
eqGam3 = Net_demand ~ s(as.numeric(Date), k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.1 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Summer_break + Christmas_break + DLS + BH_Holiday + s(toy, k=20, bs='cc') +
  + Holiday + Holiday_zone_b +  BH + BH_before + BH_after 

timeCV("gam", eqGam3, Data0, Data0$Net_demand, 4, min_train_size=633)

#plot des résidus + hypothèse gaussienne/QQplot ?
modGam = gam(eqGam3, data=Data0)
resGam = residuals(modGam)
qqnorm(resGam)
qqline(resGam)

plot(resGam~Data0$Date)

################################################################################
#                             GAM + RANDOM FOREST                              #
################################################################################

# analyse ascendante 
eqGamRf1 = Net_demand ~ s(Time,k=3, bs='cr') + te(Load.1, Load.7) + s(Solar_power.7, bs='cr') + s(Wind_power.1, bs='cr') + s(Net_demand.1) + WeekDays3 + Covid +
  s(Nebulosity_weighted) +  s(Wind_weighted) + te(Temp, Temp_s95) + Season + Summer_break + Christmas_break + DLS + BH_Holiday

eqGamRf2 = Net_demand ~ s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.7 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Season + Summer_break + Christmas_break + DLS + BH_Holiday 

eqGamRf3 = Net_demand ~ s(Time, k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  Solar_power.7 + Wind_power.1 + Net_demand.1 + Net_demand.7 + WeekDays3 + Covid +
  Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  Season + Summer_break + Christmas_break + DLS + BH_Holiday + s(toy, k=30, bs='cc') +
  + Holiday + Holiday_zone_a + Holiday_zone_b + Holiday_zone_c + BH + BH_before + BH_after

# gamrf4
eqGamRf4 = Net_demand~s(as.numeric(Date),k=3, bs='cr') + 
  te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') + 
  Solar_power.1 + Wind_power.1 + WeekDays3 + Covid+ Nebulosity_weighted + 
  Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + Summer_break +
  DLS + BH_Holiday + BH_before + s(toy,k=30, bs='cc')

eqRes4 = residuals ~ Time + Load.1 + Load.7 + Net_demand.1 + Net_demand.7 +
  Solar_power.1 + Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + 
  Wind_weighted + Temp + Temp_s95 + Summer_break + DLS + BH_Holiday + 
  BH_before + toy

timeCV("gamrf", eqGamRf4, Data0, Data0$Net_demand, 4, min_train_size=633, eqRes=eqRes4)

# gamrf5
eqGamRf5 = Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') +
  te(Net_demand.1, Net_demand.7, bs='cr') + te(Solar_power.1, Wind_power.1, bs='cr') +
  WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + 
  te(Temp, Temp_s95, k=3, bs='cr') + BH_Holiday + BH_before + s(toy,k=30, bs='cc') +
  Temp_trunc2 

eqRes5 = residuals ~ Time + Load.1 + Load.7 + Net_demand.1 + Net_demand.7 +
  Solar_power.1 + Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + 
  Wind_weighted + Temp + Temp_s95 + BH_Holiday + BH_before + toy + Temp_trunc2

timeCV("gamrf", eqGamRf5, Data0, Data0$Net_demand, 4, min_train_size=633, eqRes=eqRes5)

# 123 PL public/125 PL private (final score)
eqGam123 <- Net_demand~s(as.numeric(Date),k=3, bs='cr') + te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr') + 
  te(Solar_power.1, Wind_power.1, bs='cr') + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + Temp_trunc2

eqRes123 = residuals ~ Time + toy + Temp + Temp_s99 + Temp_s95_max + Temp_s99_max + Summer_break + Christmas_break +
  Nebulosity + Wind + Solar_power.7 + Wind_power.7 + Temp_s95_min + Temp_s99_min + DLS +
  Inflation + Covid + Price + Season + WeekDays3 + BH 

timeCV("gamrf", eqGam123, Data0, Data0$Net_demand, 4, min_train_size=633, eqRes=eqRes123)

# Importance des variables :
res = residuals(gam(eqGam123, data=Data0))

eqResComp = residuals ~ Net_demand.1 + Net_demand.7 + Load.1 + Load.7 + Solar_power.1 + Solar_power.7 + Wind_power.1 +
  Wind_power.7 + Wind + Wind_weighted + Nebulosity + Nebulosity_weighted + Temp + Temp_s95 + Temp_s99 +
  Temp_s95_min + Temp_s95_max + Temp_s99_min + Temp_s99_max + DLS + BH_before + BH_after + BH + Holiday + 
  Holiday_zone_a + Holiday_zone_b + Holiday_zone_c + Temp_trunc1 + Temp_trunc2 + DLS + Christmas_break + Summer_break + Covid +
  Inflation + Price
  
model_rf = ranger(eqResComp, data=cbind(Data0, res), importance='permutation')
varimp = sort(model_rf$variable.importance)
varimp
plot(1:length(varimp), varimp, main="Variable.importance on residuals", ylab="importance")
text(1:length(varimp), varimp, labels=names(varimp), pos = 1, cex = 0.5)

# tests autres gams
eq = Net_demand~te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr', k=c(3, 3)) + 
  te(Solar_power.1, Wind_power.1, bs='cr', k=c(3, 3)) + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + Temp_trunc2 +
  te(as.numeric(Date), Temp, k=c(3,5)) + te(Temp_s95_min, Temp_s95_max) + DLS + BH_after + Christmas_break + 
  Holiday_zone_b

eq = Net_demand~te(Load.1, Load.7, bs='cr') + te(Net_demand.1, Net_demand.7, bs='cr', k=c(3, 3)) + 
  te(Solar_power.1, Wind_power.1, bs='cr', k=c(3, 3)) + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + te(Temp, Temp_s95, k=3, bs='cr') + 
  BH_Holiday + BH_before + s(toy,k=30, bs='cc') + Temp_trunc2 + 
  te(as.numeric(Date), Temp, k=c(3,5))


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
  BH_Holiday + BH_before + s(toy,k=35, bs='cc') + Temp_trunc2 + Season + s(Wind, k=6)

eq13res = residuals~ Time + Load.1 + Load.7 + Net_demand.1 + Net_demand.7 + 
  Solar_power.1 + Wind_power.1 + WeekDays3 + Covid + Nebulosity_weighted + Wind_weighted + Temp + Temp_s95 + 
  BH_Holiday + BH_before + toy + Temp_trunc2 + Season + Wind

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


################################################################################
#                              GAM + BOOSTING                                  #
################################################################################







################################################################################
#                                     QGAM                                     #
################################################################################








