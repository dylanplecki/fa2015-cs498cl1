
wine_data<-read.csv('wine.csv', header=FALSE)
wine_cov_matrix<-cov(winedata)
wine_eigen<-eigen(wine_cov_matrix)
wine_e_vectors<-wine_eigen$vectors[,order(wine_eigen$values, decreasing = TRUE)]
plot_value<-wine_eigen$vectors[,1:3]
test<-t(plot_value)

plot(test[1,], main="10.5.B: First Three Principle Components",
     xlab="", ylab = "", col="red", ylim = c(min(test), max(test)))
lines(test[1,], type="h", col="red")

points(test[2,], col="green")
lines(test[2,], type="h", col="green")

points(test[3,], col="blue")
lines(test[3,], type="h", col="blue")

legend('bottomright', c("1st PC", "2nd PC", "3rd PC"), 
       lty=1, col=c('red', 'green', 'blue'), bty='n', cex=.75)

box()
