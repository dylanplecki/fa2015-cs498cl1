kw<-read.table('kittiwak.txt', header=TRUE)

kdat1<-data.frame(Area=kw$Area, Population=kw$Population)
kdat2<-data.frame(Area=kw$Area, Population=log(kw$Population))
kdat3<-data.frame(Area=log(kw$Area), Population=kw$Population)
kdat4<-data.frame(Area=log(kw$Area), Population=log(kw$Population))

kdat1.lm<-lm(Population ~ Area, data=kdat1)
kdat2.lm<-lm(Population ~ Area, data=kdat2)
kdat3.lm<-lm(Population ~ Area, data=kdat3)
kdat4.lm<-lm(Population ~ Area, data=kdat4)

options(digits=3)

par(mfrow=c(2,2))

plot(kdat1, col='blue', xlab=NA, ylab=NA)
abline(kdat1.lm, col='red', lwd=1.5)
title(bquote(bold('Kittiwake Colonies')~'(R'['adj']^2~'='~.(summary(kdat1.lm)$adj.r.squared)*')'),
      xlab=expression(paste("Area (km"^"2"~")")), ylab="Population")
legend('bottomright', c("Actual", "Predicted"),
       lwd=c(1, 1.5),col=c('blue','red'),
       lty=c(NA, 1), pch=c(1, NA))

plot(kdat2, col='blue', xlab=NA, ylab=NA)
abline(kdat2.lm, col='red', lwd=1.5, untf=TRUE)
title(bquote(bold('Kittiwake Colonies')~'(R'['adj']^2~'='~.(summary(kdat2.lm)$adj.r.squared)*')'),
      xlab=expression(paste("Area (km"^"2"~")")), ylab="log(Population)")
legend('bottomright', c("Actual", "Predicted"),
       lwd=c(1, 1.5),col=c('blue','red'),
       lty=c(NA, 1), pch=c(1, NA))

plot(kdat3, col='blue', xlab=NA, ylab=NA)
abline(kdat3.lm, col='red', lwd=1.5)
title(bquote(bold('Kittiwake Colonies')~'(R'['adj']^2~'='~.(summary(kdat3.lm)$adj.r.squared)*')'),
      xlab=expression("log(Area) (km"^"2"~")"), ylab="Population")
legend('topleft', c("Actual", "Predicted"),
       lwd=c(1, 1.5),col=c('blue','red'),
       lty=c(NA, 1), pch=c(1, NA))

plot(kdat4, col='blue', xlab=NA, ylab=NA)
abline(kdat4.lm, col='red', lwd=1.5)
title(bquote(bold('Kittiwake Colonies')~'(R'['adj']^2~'='~.(summary(kdat4.lm)$adj.r.squared)*')'),
      xlab=expression(paste("log(Area) (km"^"2"~")")), ylab="log(Population)")
legend('topleft', c("Actual", "Predicted"),
       lwd=c(1, 1.5),col=c('blue','red'),
       lty=c(NA, 1), pch=c(1, NA))
