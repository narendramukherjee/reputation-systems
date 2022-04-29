library(R.utils)
library(data.table)

s <- fread("posterior_samples_DF.gz")
s[,mean_rho0:=mean(rho_0), by = .(asin)][,mean_rho1:=mean(rho_1), by = .(asin)][,mean_hp:=mean(h_p), by = .(asin)]
s[,ymin_rho0:=quantile(rho_0, 0.25), by = .(asin)][,ymax_rho0:=quantile(rho_0, 0.75), by = .(asin)]
s[,ymin_rho1:=quantile(rho_1, 0.25), by = .(asin)][,ymax_rho1:=quantile(rho_1, 0.75), by = .(asin)]
s[,ymin_hp:=quantile(h_p, 0.25), by = .(asin)][,ymax_hp:=quantile(h_p, 0.75), by = .(asin)]

s_uniq <- unique(s, by = c("asin"))
#s_uniq <- s_uniq[1]

su1 <- s_uniq[mean_rho0<2]

s_uniq$asin <- factor(s_uniq$asin)

p1 <- ggplot(s_uniq, aes(x = asin,y =mean_rho0)) + geom_point() + geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0)) +
  #theme(axis.title.x=element_blank(),
      #  axis.text.x=element_blank(),
      #  axis.ticks.x=element_blank()) +
  coord_flip() + #scale_y_reverse() +
  theme(axis.title.y=element_blank(),
   axis.text.y=element_blank(),
   axis.ticks.y=element_blank()) 

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

geom_errorbar(aes(ymin = ymin_rho0, ymax = ymax_rho0))



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
