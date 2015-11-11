
seeds.data<-read.csv('seeds.csv', header=TRUE)
seeds.data.covm<-cov(seeds.data)
seeds.data.e<-eigen(seeds.data.covm)
seeds.data.ev<-sort(seeds.data.e$values, decreasing = TRUE)

plot(seeds.data.ev, type="n", main="10.6.B: Eigen Values of Seed List",
     ylab = "Eigen Value", xlab = "Number of Eigen Value (sorted)")
lines(seeds.data.ev, type="o")
