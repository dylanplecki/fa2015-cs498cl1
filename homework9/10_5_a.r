
wine_data<-read.csv('wine.csv', header=FALSE)
wine_cov_matrix<-cov(scale(winedata))
wine_eigen<-eigen(wine_cov_matrix)
wine_eigen_values<-sort(wine_eigen$values, decreasing = TRUE)

plot(wine_eigen_values, type="n", main="10.5.A: Eigen Values of Wine List",
     ylab = "Eigen Value (normalized)", xlab = "Number of Eigen Value (sorted)")
lines(wine_eigen_values, type="o")
