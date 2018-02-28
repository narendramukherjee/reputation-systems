###########################
### With tablet dataset ###
###########################
library(data.table)
library(ggplot2)
library(plyr)
library(reshape2)
library(lubridate)
library(broom)

setwd('/home/shrabastee/Dropbox/project_shrabastee_amin/tablet_dataset/')
prod <- fread('Market Dynamics of Product.csv')

# Reading in saved data

main_data <- readRDS('main_data.Rds')
prod <- fread('Market Dynamics of Product.csv')
prod <- prod[,.(Item_ID,Title)]
main_data <- merge(main_data, prod, by = c("Item_ID"))

# Split main data into 4 data frames: sales rank, price, rev summary and characteristics

main_data1 <- main_data[,c(1:25)]
main_data2 <- main_data[,c(1, 26:28)]
main_data3 <- main_data[,c(1, 29:52)]
main_data4 <- main_data[,c(1, 53:63)]
main_data4 <- data.table(main_data4)
#main_data4 <- main_data4[,.(Item_ID, Item_Weight, Screen_Size, Average_Battery_Life__in_hours_, RAM, Title)]
main_data4$Screen_Size <- as.numeric(gsub(" inches", "", main_data4$Screen_Size))
main_data4$Average_Battery_Life__in_hours_ <- as.numeric(gsub(" hours", "", main_data4$Average_Battery_Life__in_hours_))
main_data4$Item_Weight <- gsub(" pounds", "", main_data4$Item_Weight)
main_data4$Item_Weight <- gsub(" oz", "", main_data4$Item_Weight)
main_data4$Item_Weight <- as.numeric(main_data4$Item_Weight)
main_data4$Item_Weight[which(main_data4$Item_Weight>=5)] <- main_data4$Item_Weight[which(main_data4$Item_Weight>=5)]/16 # convert oz to pounds
main_data4$RAM <- gsub(" MB", "", main_data4$RAM)
main_data4$RAM <- gsub(" GB.*", "", main_data4$RAM)
main_data4$RAM <- gsub(" D.*", "", main_data4$RAM)
main_data4$RAM <- as.numeric(main_data4$RAM)
main_data4$RAM[which(main_data4$RAM<=5)] <- main_data4$RAM[which(main_data4$RAM<=5)]*1000




# Putting dataset together

main_data_long1 <- melt(main_data1, id.vars = c("Item_ID"), measure.vars = c(paste0("Sales_rank_week",1:24)))
main_data_long3 <- melt(main_data3, id.vars = c("Item_ID"), measure.vars = c(paste0("Lowest_new_price_week",1:24)))
main_data_long1 <- main_data_long1[order(main_data_long1$Item_ID),]
main_data_long3 <- main_data_long3[order(main_data_long3$Item_ID),]
setnames(main_data_long1, c("variable", "value"), c("Sales Rank Week", "SalesRank"))
setnames(main_data_long3, c("variable", "value"), c("Price Week", "Price"))


main_data_long <- cbind(main_data_long1,main_data_long3)
main_data_long$Item_ID <- NULL
main_data_long <- merge(main_data_long,main_data2, by = c("Item_ID"))
main_data_long <- merge(main_data_long,main_data4, by = c("Item_ID"))

main_data_long$Price <- as.numeric(gsub("\\$", "", main_data_long$Price))
hist(main_data_long$Price)
main_data_long <-na.omit(main_data_long)

main_data_long <- data.table(main_data_long)
main_data_long <- main_data_long[,total_rev:=30501]
main_data_long <- main_data_long[,ratio:=num_rev/total_rev, by = .(Item_ID)]
main_data_long <- main_data_long[,sales_rank_sd:=sd(SalesRank), by = .(Item_ID)]
main_data_long <- main_data_long[,sd_price:=sd(Price), by = .(Item_ID)]

# Collapse categorical variables

main_data_long$OS <- ""
main_data_long$OS[grep("Android", main_data_long$Operating_System)] <- "Android"
main_data_long$OS[grep("Windows", main_data_long$Operating_System)] <- "Windows"
main_data_long$OS[grep("Blackberry", main_data_long$Operating_System)] <- "Blackberry"
main_data_long$OS[grep("Apple", main_data_long$Operating_System)] <- "Apple"
main_data_long$OS[which(main_data_long$OS=="")] <- "Others"

main_data_long$Screen_Resolution[which(main_data_long$Screen_Resolution %in% c("480x800","800x480","800X480"))] <- "800x480"
main_data_long$Screen_Resolution[which(main_data_long$Screen_Resolution %in% c("1336x768"))] <- "1366x768"


# Regressions

reg1 <-lm(log(ratio) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ + factor(OS) + factor(Screen_Resolution) + Item_Weight, data = main_data_long, offset = avg_rev)
reg1_tidy <- tidy(clx(reg1,1,main_data_long$Item_ID))
reg1_rsq <- summary(reg1)$adj.r.squared
reg2 <-lm(log(ratio) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ + factor(OS) + factor(Screen_Resolution), data = main_data_long, offset = avg_rev)
reg2_tidy <- tidy(clx(reg2,1,main_data_long$Item_ID))
reg2_rsq <- summary(reg2)$adj.r.squared
reg3 <-lm(log(ratio) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ + factor(OS), data = main_data_long, offset = avg_rev)
reg3_tidy <- tidy(clx(reg3,1,main_data_long$Item_ID))
reg3_rsq <- summary(reg3)$adj.r.squared
reg4 <-lm(log(ratio) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_, data = main_data_long, offset = avg_rev)
reg4_tidy <- tidy(clx(reg4,1,main_data_long$Item_ID))
reg4_rsq <- summary(reg4)$adj.r.squared 
reg5 <-lm(log(ratio) ~ log(Price) + Average_Battery_Life__in_hours_ + Screen_Size + RAM, data = main_data_long, offset = avg_rev) #17

reg5_tidy <- tidy(clx(reg5,1,main_data_long$Item_ID))
reg5_rsq <- summary(reg5)$adj.r.squared  
reg5_tidy$estimate <- round(reg5_tidy$estimate, 3)
  
#pooled_ols1 <- lm(-log(SalesRank) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ , data = main_data_long, offset = avg_rev)
pooled_ols2 <-lm(log(ratio) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ + factor(Operating_System) + factor(Screen_Resolution) + Item_Weight, data = main_data_long, offset = avg_rev)
#s1 <- summary(pooled_ols1)
s2 <- summary(pooled_ols2)
s2
s1
#library(sandwich)
#s2$coefficients[, 2] <- sqrt(diag(vcovHC(pooled_ols2)))
tidied_pooled2 <- tidy(pooled_ols2) # heterosked consistent
#tidied_pooled1 <- tidy(pooled_ols1) # heterosked consistent

s2
# clustered

clx <-function(fm, dfcw, cluster){
  library(sandwich)
  library(lmtest)
  M <- length(unique(cluster))
  N <- length(cluster)
  dfc <- (M/(M-1))*((N-1)/(N-fm$rank))
  u <- apply(estfun(fm),2,
             function(x) tapply(x, cluster, sum))
  vcovCL <- dfc*sandwich(fm, meat=crossprod(u)/N)*dfcw
  coeftest(fm, vcovCL) }

#pooled_ols <- lm(-log(SalesRank) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ , data = main_data_long, offset = avg_rev)
#summary(pooled_ols)
clx(pooled_ols1,1,main_data_long$Item_ID) # price, ram, avg battery life, operating system, screen res.

# Try LASSO for variable selection

library(glmnet)
formula1 <- log(ratio) ~ Price + Screen_Size + RAM + Average_Battery_Life__in_hours_ + factor(Operating_System) + factor(Screen_Resolution) + Item_Weight
model1 <- sparse.model.matrix(formula1, data = main_data_long)
s1 <- summary(pooled_ols1)
cv1 <- cv.glmnet(model1,log(main_data_long$ratio), family = "gaussian", nfold = 10, type.measure = "deviance", alpha = 1, offset = main_data_long$avg_rev)
md1 <- glmnet(model1,log(main_data_long$ratio), family = "gaussian", lambda = cv1$lambda.1se, alpha = 1, offset = main_data_long$avg_rev)
coef(md1)


# Evolution of sales rank not important here: let's take averages (same estimates)
main_data_long <- data.table(main_data_long)
main_data_long <- main_data_long[,mean_sales_rank:=mean(SalesRank), by = .(Item_ID)]
main_data_long <- main_data_long[,mean_price:=mean(Price), by = .(Item_ID)]
main_data_undup <- unique(main_data_long, by = c("Item_ID"))
main_data_undup <- main_data_undup[,total_rev:=sum(num_rev)]
main_data_undup <- main_data_undup[,ratio:=num_rev/total_rev, by = .(Item_ID)]
ols <- lm(-log(mean_sales_rank) ~ mean_price + Screen_Size + RAM + Average_Battery_Life__in_hours_, data = main_data_undup, offset = avg_rev)
summary(ols)
tidied_ols <- tidy(ols)
ols2 <- lm(log(ratio) ~ mean_price + Screen_Size + RAM + Average_Battery_Life__in_hours_, data = main_data_undup, offset = avg_rev)
summary(ols2)
tidied_ols2 <- tidy(ols2)
tidied_ols2$estimate <- round(tidied_ols2$estimate, 3)

# Branded vs unbranded products

unbranded <- c("Le Pan TC 970 9.7-Inch Multi-Touch LCD Google Android Tablet PC", "VIZIO 8-Inch Tablet with WiFi - VTAB1008")
u_id <- c("B004PGMFG2","B005B9G79I")
unbranded <- data.frame(cbind(unbranded,u_id))
branded <- c("Apple iPad (first generation) MB292LL/A Tablet (16GB, Wifi)", "HP TouchPad Wi-Fi 32 GB 9.7-Inch Tablet Computer")
b_id <- c("B002C7481G","B0055D66V4")
branded <- data.frame(cbind(branded,b_id))

# Just one product

rev <- readRDS('reviews_cleaned.Rds')
rev <- rev[,.(Item_ID, Rating,Review_date)]
rev <- rev[,.std:=sd(Rating), by = .(Item_ID)]
rev <- rev[,avg:=mean(Rating), by = .(Item_ID)]
rev <- rev[Item_ID %in% branded$b_id| Item_ID %in% unbranded$u_id,]
rev <- unique(rev, by = c("Item_ID"))
rev <- rev[Item_ID==unbranded$u_id[2],]
rev$avg <- 
  rev <- rev[order(Review_date),]
rev$Review_date <- NULL
write.table(rev, "vizio_time_series.txt",row.names = FALSE, col.names = TRUE,quote = FALSE, sep = "\t")

# parameters

main_data_undup <- data.table(main_data_undup)
params <- main_data_undup[Item_ID %in% branded$b_id| Item_ID %in% unbranded$u_id,]