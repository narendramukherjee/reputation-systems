setwd("/home/sbanerjee/researchdrive/TiSEM0004/SNPE/")

library(R.utils)
library(data.table)

s <- fread("posterior_samples_DF.gz")
s[,mean_rho0:=mean(rho_0), by = .(asin)][,mean_rho1:=mean(rho_1), by = .(asin)][,mean_hp:=mean(h_p), by = .(asin)]
s[,sd_rho0:=sd(rho_0), by = .(asin)][,sd_rho1:=sd(rho_1), by = .(asin)][,sd_hp:=sd(h_p), by = .(asin)]
s[,ymin_rho0:=quantile(rho_0, 0.25), by = .(asin)][,ymax_rho0:=quantile(rho_0, 0.75), by = .(asin)]
s[,ymin_rho1:=quantile(rho_1, 0.25), by = .(asin)][,ymax_rho1:=quantile(rho_1, 0.75), by = .(asin)]
s[,ymin_hp:=quantile(h_p, 0.25), by = .(asin)][,ymax_hp:=quantile(h_p, 0.75), by = .(asin)]

s_uniq <- unique(s, by = c("asin"))
#s_uniq[,vol_quant:=cut(s_uniq$num_reviews,4)]
#s_uniq <- s_uniq[1]

su1 <- s_uniq[mean_rho0<2]

s_uniq$asin <- factor(s_uniq$asin)
s_uniq <- s_uniq  %>%
  mutate(quantile = ntile(num_reviews, 4))
s_uniq$quantile <- as.factor(s_uniq$quantile)
s_uniq <- s_uniq  %>%
  mutate(hp_quantile = ntile(mean_hp, 4))
s_uniq$hp_quantile <- as.factor(s_uniq$hp_quantile)


p1h <- ggplot(s_uniq, aes(x = asin,y =mean_rho0)) + geom_point() + 
  geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0)) +
  facet_wrap(hp_quantile~., scales = "free") +
  #coord_flip() +scale_y_reverse() +
  theme(axis.title.x=element_blank(),
   axis.text.x=element_blank(),
   axis.ticks.x=element_blank())
 # +
  
p1h

hp_labs <- c("1","2","3","4")
names(hp_labs) <- c("0-25th percentile","25th to 50th percentile","50th to 75th percentile","75th to 100th percentile" )

p2h <- ggplot(s_uniq, aes(x = asin,y =mean_rho1)) + geom_point() + 
  geom_errorbar(aes(ymin = ymin_rho1, ymax = ymax_rho1)) +
  facet_wrap(hp_quantile~., scales = "free", labeller = hp_labs) +
  #coord_flip() +scale_y_reverse() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + ylab("Means and 75% credible intervals for each product")


p2h


p1 <- ggplot(s_uniq, aes(x = asin,y =mean_rho0)) + geom_point() + geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0)) +
  #theme(axis.title.x=element_blank(),
      #  axis.text.x=element_blank(),
      #  axis.ticks.x=element_blank()) +
  coord_flip() + #scale_y_reverse() +
  theme(axis.title.y=element_blank(),
   axis.text.y=element_blank(),
   axis.ticks.y=element_blank())

p1_vol0 <- ggplot(s_uniq, aes(x = quantile,y =mean_rho0)) + geom_boxplot(notch=FALSE)
p1_vol
p1_vol1 <- ggplot(s_uniq, aes(x = quantile,y =mean_rho1)) + geom_boxplot(notch=FALSE)
p1_hp <- ggplot(s_uniq, aes(x = quantile,y =mean_hp)) + geom_boxplot(notch=FALSE)
p1_hp

p2 <- ggplot(s_uniq, aes(x = asin,y =mean_rho1)) + geom_point() + geom_errorbar(aes(ymin = ymin_rho1, ymax = ymax_rho1)) +
  #theme(axis.title.x=element_blank(),
  #  axis.text.x=element_blank(),
  #  axis.ticks.x=element_blank()) +
  coord_flip() + #scale_y_reverse() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) 

p3 <- ggplot(s_uniq, aes(x = asin,y =mean_hp)) + geom_point() + geom_errorbar(aes(ymin = ymin_hp, ymax = ymax_hp)) +
  #theme(axis.title.x=element_blank(),
  #  axis.text.x=element_blank(),
  #  axis.ticks.x=element_blank()) +
  coord_flip() + #+ scale_y_reverse() +
 theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) 

library(gridExtra)
library(cowplot)
plot_grid(p1, p2, p3, labels=c("rho_0", "rho_1", "h_p"), ncol = 2, nrow = 2)

# variance: TBD

# rho+ and rho- for different values of hp
# product characteristics

prods <- fread("products_with_avg_price.txt")
prods <- prods[external_id %in% s_uniq$asin]
prods$external_id <- as.factor(prods$external_id)
s_uniq <- merge(s_uniq,prods, by.x = "asin", by.y = "external_id")
s_uniq$avg_price <- as.numeric(s_uniq$avg_price)

s_uniq <- s_uniq  %>%
  mutate(p_quants = ntile(avg_price, 4))
s_uniq$p_quants <- as.factor(s_uniq$p_quants)

ggplot(s_uniq, aes(x = p_quants,y =mean_rho0)) + geom_boxplot(notch=FALSE)
ggplot(s_uniq, aes(x = p_quants,y =mean_rho1)) + geom_boxplot(notch=FALSE)
ggplot(s_uniq, aes(x = p_quants,y =mean_hp)) + geom_boxplot(notch=FALSE)

summary(lm(mean_rho0 ~ log(avg_price), data = s_uniq))
summary(lm(mean_rho1 ~ log(avg_price), data = s_uniq))
summary(lm(mean_hp ~ log(avg_price), data = s_uniq))


geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0))
counts_b <- s_uniq %>% count(brand)
counts_b <- counts_b[n>=20]
s_uniq[,branded:=ifelse(s_uniq$brand %in% counts_b$brand,1,0)]

summary(lm(mean_rho0 ~ branded, data = s_uniq))
summary(lm(mean_rho1 ~ branded, data = s_uniq))
summary(lm(mean_hp ~ branded, data = s_uniq))

summary(lm(sd_rho0 ~ log(num_reviews), data = s_uniq))
summary(lm(sd_rho1 ~ log(num_reviews), data = s_uniq))
summary(lm(sd_hp ~ log(num_reviews), data = s_uniq))

summary(lm(mean_rho0 ~ log(num_reviews), data = s_uniq))
summary(lm(mean_rho1 ~ log(num_reviews), data = s_uniq))
summary(lm(mean_hp ~ log(num_reviews), data = s_uniq))

ggplot(all, aes(x = outcome,y =ate)) + 
  geom_pointrange(aes(ymin = ci_lower, ymax = ci_upper)) +
  facet_wrap(.~model) + theme_bw() +
  geom_hline(yintercept=0,color = "red")

geom_boxplot(outlier.colour="red", outlier.shape=8,
             outlier.size=4) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())


main_plot <- ggplot(s, aes(x = DependentVar,y =point_estimate, color = DependentVar)) + 
  geom_pointrange(aes(ymin = ci_lower, ymax = ci_upper)) + 
  facet_wrap(Outcome~., scales = "free", labeller = outcomes) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  geom_hline(yintercept=0, color = "red")+
  theme(legend.key.size = unit(1, "cm")) +
  theme(legend.title = element_text(size = 8), 
        legend.text = element_text(size = 8)) +
  labs(y="Point Estimate",  
       col="Device Type") 
device_plot
