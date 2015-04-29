D = read.table('equal_tailed_lengths.csv', sep=',')
mu.vals = D[,1]
ci.length = D[,2]

pdf("ciBivar.pdf",height=4,width=4)
par(mar=c(4.1,4.1,3.1,1.1))
plot(mu.vals,ci.length,type="l",ylim=c(0,4.5),
     xlab=expression(mu),
     ylab="Interval Length",
     main="Expected CI Length",
     yaxt="n")
axis(2,0:4)
abline(2*qnorm(0.975),0,lty=2)
legend("bottomleft",legend=c("Data Splitting","Data Carving"),lty=2:1,bty="n")
dev.off()