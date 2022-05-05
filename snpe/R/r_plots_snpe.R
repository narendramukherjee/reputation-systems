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
s_uniq$asin <- factor(s_uniq$asin)

# product characteristics

prods <- fread("products_with_avg_price.txt")
prods <- prods[external_id %in% s_uniq$asin]
prods$external_id <- as.factor(prods$external_id)
s_uniq <- merge(s_uniq,prods, by.x = "asin", by.y = "external_id")
s_uniq$avg_price <- as.numeric(s_uniq$avg_price)

high_plus <- s_uniq[mean_rho1>mean(mean_rho1)]
high_minus <- s_uniq[mean_rho0>mean(mean_rho0)]


# Create volume and herding quantiles

s_uniq <- s_uniq  %>%
  mutate(quantile = ntile(num_reviews, 4))
s_uniq$quantile <- as.factor(s_uniq$quantile)
s_uniq <- s_uniq  %>%
  mutate(hp_quantile = ntile(mean_hp, 4))
s_uniq$hp_quantile <- as.factor(s_uniq$hp_quantile)

# Herding plots

hp_labs <- c("1"="Herding parameter h_p in 0-25th percentile",
             "2" = "Herding parameter h_p in 25th to 50th percentile",
             "3"= "Herding parameter h_p in 50th to 75th percentile",
             "4" = "Herding parameter h_p in 75th to 100th percentile")

p1h <- ggplot(s_uniq, aes(x = asin,y =mean_rho0)) + geom_point() + 
  geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0)) +
  facet_wrap(hp_quantile~., scales = "free", labeller = as_labeller(hp_labs)) +
  #coord_flip() +scale_y_reverse() +
  theme(axis.title.x=element_blank(),
   axis.text.x=element_blank(),
   axis.ticks.x=element_blank()) + ylab("Means and IQR of posterior for each product (\u03c1-)")
 # +
  
p1h



p2h <- ggplot(s_uniq, aes(x = asin,y =mean_rho1)) + geom_point() + 
  geom_errorbar(aes(ymin = ymin_rho1, ymax = ymax_rho1)) +
  facet_wrap(hp_quantile~., scales = "free", labeller = as_labeller(hp_labs)) +
  #coord_flip() +scale_y_reverse() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + ylab("Means and IQR of posterior for each product (\u03c1+)")


p2h


p1 <- ggplot(s_uniq, aes(x = asin,y =mean_rho0)) + geom_point() + geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0)) +
  #theme(axis.title.x=element_blank(),
      #  axis.text.x=element_blank(),
      #  axis.ticks.x=element_blank()) +
  coord_flip() + #scale_y_reverse() +
  theme(axis.title.y=element_blank(),
   axis.text.y=element_blank(),
   axis.ticks.y=element_blank()) + ylab("Means and IQR of posterior for each product (\u03c1-)")
p1

p1_vol0 <- ggplot(s_uniq, aes(x = quantile,y =mean_rho0)) + geom_boxplot(notch=FALSE) + xlab("Review volume quantile")+ ylab("Mean \u03c1-")
p1_vol0
p1_vol1 <- ggplot(s_uniq, aes(x = quantile,y =mean_rho1)) + geom_boxplot(notch=FALSE) + xlab("Review volume quantile") + ylab("Mean \u03c1+")
p1_vol1

s_uniq_673 <- s_uniq[num_reviews!=673]

ggplot(s_uniq_673, aes(x = num_reviews,y =mean_rho0)) + geom_point() +
  geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0)) + xlab("Review volume quantile")+ ylab("Mean \u03c1-") + theme_bw()


ggplot(s_uniq_673, aes(x = num_reviews,y =mean_rho1)) + geom_point() +
  geom_errorbar(aes(ymin = ymin_rho1, ymax = ymax_rho1)) + xlab("Review volume quantile")+ ylab("Mean \u03c1+") + theme_bw()


ggplot(s_uniq_673, aes(x = num_reviews,y =mean_hp)) + geom_point() +
  geom_errorbar(aes(ymin = ymin_hp, ymax = ymax_hp)) + xlab("Review volume quantile")+ ylab("Mean h_p") + theme_bw()

p1_hp <- ggplot(s_uniq, aes(x = quantile,y =mean_hp)) + geom_boxplot(notch=FALSE)
p1_hp

p2 <- ggplot(s_uniq, aes(x = asin,y =mean_rho1)) + geom_point() + geom_errorbar(aes(ymin = ymin_rho1, ymax = ymax_rho1)) +
  #theme(axis.title.x=element_blank(),
  #  axis.text.x=element_blank(),
  #  axis.ticks.x=element_blank()) +
  coord_flip() + #scale_y_reverse() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) + ylab("Means and IQR of posterior for each product (\u03c1+)")

p3 <- ggplot(s_uniq, aes(x = asin,y =mean_hp)) + geom_point() + geom_errorbar(aes(ymin = ymin_hp, ymax = ymax_hp)) +
  #theme(axis.title.x=element_blank(),
  #  axis.text.x=element_blank(),
  #  axis.ticks.x=element_blank()) +
  coord_flip() + #+ scale_y_reverse() +
 theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) + ylab("Means and IQR of posterior for each product (h_p)")

library(gridExtra)
library(cowplot)
plot_grid(p1, p2, p3, labels=c("rho_0", "rho_1", "h_p"), ncol = 2, nrow = 2)

# variance: TBD

# rho+ and rho- for different values of hp


s_uniq <- s_uniq  %>%
  mutate(p_quants = ntile(avg_price, 4))
s_uniq$p_quants <- as.factor(s_uniq$p_quants)

ggplot(s_uniq, aes(x = p_quants,y =mean_rho0)) + geom_boxplot(notch=FALSE)
ggplot(s_uniq, aes(x = p_quants,y =mean_rho1)) + geom_boxplot(notch=FALSE)
ggplot(s_uniq, aes(x = p_quants,y =mean_hp)) + geom_boxplot(notch=FALSE)



summary(s1 <- lm(mean_rho0 ~ log(avg_price), data = s_uniq_673))
summary(s2 <- lm(mean_rho1 ~ log(avg_price), data = s_uniq_673))
summary(s3 <- lm(mean_hp ~ log(avg_price), data = s_uniq_673))

summary(s1 <- lm(mean_rho0 ~ log(avg_price) + num_reviews, data = s_uniq_673))
summary(s2 <- lm(mean_rho1 ~ log(avg_price) + num_reviews, data = s_uniq_673))
summary(s3 <- lm(mean_hp ~ log(avg_price) + num_reviews, data = s_uniq_673))

stargazer(s1,s2,s3, type = "html")




geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0))
counts_b <- s_uniq %>% count(brand)
counts_b <- counts_b[n>=1]
s_uniq[,branded:=ifelse(s_uniq$brand=="Unbranded"|s_uniq$brand=="None"|s_uniq$brand=="Argos Value Range",0,1)]
s_uniq_673 <- s_uniq[num_reviews!=673]

summary(t1 <- lm(mean_rho0 ~ branded + num_reviews, data = s_uniq_673))
summary(t2 <- lm(mean_rho1 ~ branded + num_reviews, data = s_uniq_673))
summary(t3 <- lm(mean_hp ~ branded + num_reviews, data = s_uniq_673))

summary(t1 <- lm(mean_rho0 ~ branded, data = s_uniq_673))
summary(t2 <- lm(mean_rho1 ~ branded, data = s_uniq_673))
summary(t3 <- lm(mean_hp ~ branded, data = s_uniq_673))

stargazer(t1,t2,t3, type = "html")


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
