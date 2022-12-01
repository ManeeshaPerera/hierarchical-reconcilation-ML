# install.packages('tsutils')
library(tsutils)

x <- matrix( rnorm(50*4,mean=0,sd=1), 50, 4) # Generate some performance statistics
x[,2] <- x[,2]+1
x[,3] <- x[,3]+0.7
x[,4] <- x[,4]+0.5
colnames(x) <- c("Method A","Method B","Method C - long name","Method D")
print(x)
nemenyi(x, plottype ='matrix')