
wine.data<-read.csv('wine.csv', header=TRUE)
wine.data.pca <- prcomp(wine.data, scale. = TRUE, center = TRUE)

library(ggbiplot)
g <- ggbiplot(wine.data.pca, obs.scale = 1, var.scale = 1, 
              ellipse = TRUE, circle = TRUE,
              groups = factor(wine.data$Cultivar, labels = c("Group 1", "Group 2", "Group 3")))
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)
