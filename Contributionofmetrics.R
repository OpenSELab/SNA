library(factoextra)
library("FactoMineR")

path <- "C:/gln/mygrap/new/data2/bb/"
files <- list.files(path=path, pattern="*.csv")
for(file in files)
{
  s1=paste(path,'/',file,sep='')
  data<-read.csv(s1)
  X <-data[,65:118]
  X<-scale(X) 
  mu = colMeans(X)
  
  Xpca = prcomp(X,scale = TRUE)
  
  nComp = 1
  Xhat = Xpca$x[,1:nComp] %*% t(Xpca$rotation[,1:nComp])
  Xhat = scale(Xhat, center = -mu, scale = FALSE)
  
  t<-Xhat[,]-X[,]
  s2=paste("C:/gln/mygrap/pca/pcs/code/1",'/',file,sep='')
  write.table(t,s2,row.names=FALSE,col.names=TRUE,sep=",")
}

for(file in files)
{
  s1=paste(path,'/',file,sep='')
  data<-read.csv(s1)
  X <-data[,66:119]
  #X<-data
  X<-scale(X) 

  var_coord_func <- function(loadings, comp.sdev){
    loadings*comp.sdev
  }
  res.pca <- prcomp(X)
  loadings <- res.pca$rotation
  sdev <- res.pca$sdev
  var.coord <- t(apply(loadings, 1, var_coord_func, sdev)) 
  head(var.coord[, 1:4])
  
  var.cos2 <- var.coord^2
  head(var.cos2[, 1:4])
  
  comp.cos2 <- apply(var.cos2, 2, sum)
  contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}
  var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))
  head(var.contrib[, 1:4])
  
  s2=paste("C:/gln/mygrap/new/pca/pcs/code/contributor",'/',file,sep='')
  write.table(var.contrib,s2,row.names=FALSE,col.names=TRUE,sep=",")
}




data<-read.csv("C:/gln/mygrap/new/data2/bb/activemq_class_SM.csv")
X <-data[,2:119]
X<-scale(X) 
#mu = colMeans(X)
var_coord_func <- function(loadings, comp.sdev){
  loadings*comp.sdev
}
res.pca <- prcomp(X)
loadings <- res.pca$rotation
sdev <- res.pca$sdev
var.coord <- t(apply(loadings, 1, var_coord_func, sdev)) 
head(var.coord[, 1:4])

var.cos2 <- var.coord^2
head(var.cos2[, 1:4])

comp.cos2 <- apply(var.cos2, 2, sum)
contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}
var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))
head(var.contrib[, 1:4])

s2="C:/gln/mygrap/pca/pcs/activemq_class_SM.csv"
write.table(var.contrib,s2,row.names=FALSE,col.names=TRUE,sep=",")

res.pca <- PCA(X, graph = FALSE)
fviz_contrib(res.pca, choice = "var", axes = 1, top = 15)
fviz_contrib(res.pca, choice = "ind", axes = 1:2, top = 10)