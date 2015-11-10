
wine_data<-read.csv('wine.csv', header=FALSE)
wine_cov_matrix<-cov(winedata)
wine_eigen<-eigen(wine_cov_matrix)
wine_eigen_values<-sort(wine_eigen$values, decreasing = TRUE)[1:3]

plot(wine_eigen_values, main="10.5.B: First Three Principle Components", xaxt="n",
     ylab = "Eigen Value (normalized)", xlab = "Number of Eigen Value (sorted)", xlim = c(1, 3))
lines(wine_eigen_values, type="h")
axis(side=1, at=seq(0, 3, by=1))
box()
