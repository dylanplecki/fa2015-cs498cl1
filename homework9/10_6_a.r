
seeds.data<-read.csv('seeds.csv', header=TRUE)
seeds.data.pca <- prcomp(seeds.data, scale. = TRUE, center = TRUE)

library(ggbiplot)
g <- ggbiplot(seeds.data.pca, obs.scale = 1, var.scale = 1, 
              ellipse = TRUE, circle = TRUE,
              groups = factor(seeds.data$Class, labels = c("Group 1", "Group 2", "Group 3")))
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)
