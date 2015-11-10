#graphics.off()
#setEPS()

# r code for scatterplot of iris data
irisdat<-read.csv('iris.csv', header=FALSE);

library('lattice')

numiris=irisdat[, c(1, 2, 3, 4)]

#pdf("irisscatterplot.pdf")
# so that I get a image file

speciesnames<-c('setosa', 'versicolor' , 'virginica')
pchr<-c(1 , 2, 3)
colr<-c('red', 'green', 'blue', 'yellow', 'orange')
ss<-expand.grid(species =1:3)
parset<-with (ss, simpleTheme(pch=pchr[species], col=colr[species]))

splom(irisdat[,c(1:4)], groups=irisdat$V5,
  par.settings=parset,
  varnames=c('Sepal \nLength', 'Sepal \nWidth', 'petal \nLength', 'Petal \nWidth'),
  key=list(text=list(speciesnames),
  points=list(pch=pchr), columns=3)
)

#dev.off()