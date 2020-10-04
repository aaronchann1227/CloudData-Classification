---
title: "p2"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


```{r}
colnames = c('x','y','label','NDAI','SD','CORR','DF','CF','BF','AF','AN')
data1 = read.table("~/documents/stat154/project2/image_data/image1.txt", col.names = colnames)
data1$image = 'data1'


data2 = read.table("~/documents/stat154/project2/image_data/image2.txt", col.names = colnames)
data2$image = 'data2'
data3 = read.table("~/documents/stat154/project2/image_data/image3.txt", col.names = colnames)
data3$image = 'data3'
allData = rbind(data1,data2,data3)
```

```{r}
summary(allData)
table(data1$label)/dim(data1)[1]
table(data2$label)/dim(data2)[1]
table(data3$label)/dim(data3)[1]
```


```{r}
library(ggplot2)
data.labeled = allData[allData$label != 0,]

ggplot(allData, aes(x=x,y=y,color = label)) + geom_point() + facet_wrap(~image)
```

We can definitely see that there are clusters for each of the classification labels, meaning that they are not iid. If you know the label of one data point, the labels of the nearby data points are more likely to have the same label.

```{r}
library(corrplot)
correlation = cor(data.labeled[,-12])
corrplot.mixed(correlation)
```


Much of the radiance readings have high correlation, which makes sense as they are simply different angles of the same picture. There is a high linear correlation between NDAI/CORR and label, also suggesting that a linear method of classification might be useful in predicting the labels. The y coordinate has a relatively high correlation with the label, although given its definition, this may be by chance. 
Based on the correlations, since the NDAI has a high correlation with the label, we can deduce that higher values of NDAI will correspond to a cloudy surface, while lower values will correspond to a clear surface.

```{r}
set.seed(123)
data1 = data.labeled[data.labeled$image == 'data1',]
data2 = data.labeled[data.labeled$image == 'data2',]
data3 = data.labeled[data.labeled$image == 'data3',]


# METHOD 1 (divide by blocks)
BLOCK_SIZE = 10
blocks = list()
images = list(data1,data2,data3)
for (i in 1:3) {
  image = images[[i]]
  x = range(image$x)
  x.coordinates = seq(x[1],x[2],BLOCK_SIZE)
  y = range(image$y)
  y.coordinates = seq(y[1],y[2],BLOCK_SIZE)
  for (x_coord in x.coordinates) {
    for (y_coord in y.coordinates) {
      blocks[[length(blocks)+1]] = c(i,x_coord,y_coord)
    }
  }
}

TEST_PROP = 0.2
VALIDATION_PROP = (1-TEST_PROP)*0.2
n = length(blocks)

random_blocks = sample(blocks)
test = random_blocks[1:as.integer(n*TEST_PROP)]
random_blocks = random_blocks[-(1:as.integer(n*TEST_PROP))]
validation = random_blocks[1:as.integer(n*VALIDATION_PROP)]


library(dplyr)
test.data = data.frame(data1[0,])
for (tBlock in test) {
  image = images[[tBlock[1]]]
  x = tBlock[2]
  y = tBlock[3]
  test.data = rbind(test.data,image[between(image$x,x,x+9) & between(image$y,y,y+9),])
}

val.data = data.frame(data1[0,])
for (tBlock in validation) {
  image = images[[tBlock[1]]]
  x = tBlock[2]
  y = tBlock[3]
  val.data = rbind(val.data,image[between(image$x,x,x+9) & between(image$y,y,y+9),])
}

train.data = setdiff(data.labeled,test.data)
train.data = setdiff(train.data, val.data)


```


```{r}
# METHOD 2 
length(data1$x)
length(data2$x)
length(data3$x)

test.data = data3
train.data = data2
val.data = data1
```



```{r}
mean(val.data$label == -1)
mean(test.data$label == -1)
```


```{r}
x = data.matrix(data.labeled[,c(-1,-2,-3,-12)])
y = data.matrix(data.labeled[,3])
library(glmnet)
model = glmnet(x,y,alpha = 1, lambda=.1)
coef(model)
```

```{r}
sqrt(summary(lm(label~.,data=data.labeled))$r.squared)
```
Looking at the lasso regression, we can see that the important features are NDAI and CORR. This is somewhat reliable since we can see that the regression coefficient of the OLS estimate is at 0.8, suggesting a somewhat strong linear correlation between the featues and the label. We also know from the above correlations that the one of the radiance readings should be sufficient to represent all of the readings. Looking at the correlations, we see that BF has the highest sum of correlations between the radiance readings, so we choose the features NDAI, CORR, and BF to represent our data.

```{r}
```


```{r}
source('crossval.R')

K = 6
data = rbind(train.data, val.data)
data$label = (data$label+1)/2
formula = as.formula(label~.)
hyperparams = list()

noLoss = function(x,y){0}
library(MLmetrics)

models = c('QDA','LDA','dtree','logistic')
model.accuracies = data.frame(matrix(ncol = K+2, nrow = 0))
colnames(model.accuracies) = c('model',1:K,"Average")
for (model in models) {
  accuracies = CVgeneric(model,data,K,LogLoss,hyperparams,formula,1)[['accuracies']]
  average = mean(accuracies)
  model.accuracies[nrow(model.accuracies)+1,]=c(model,accuracies,average)
}

# CVgeneric('kernelSVM',data[1:15000,],4,noLoss,hyperparams,formula,1)

```

```{r}
library(pROC)
library(rpart)

trueLabels = (test.data$label+1)/2
data = data[,c(-1,-2,-12)]
model.log = glm(formula,family = 'binomial', data = data)
model.log.pred = predict(model.log, test.data, type = 'response')
model.log.acc = mean(round(model.log.pred) == trueLabels)

model.lda = lda(formula, data = data)
model.lda.pred = predict(model.lda, test.data, type = 'response')
model.lda.prediction = as.numeric(model.lda.pred$class)-1
model.lda.acc = mean(round(model.lda.prediction) == trueLabels)

model.qda = qda(formula, data = data)
model.qda.pred = predict(model.qda, test.data, type = 'response')
model.qda.prediction = as.numeric(model.qda.pred$class)-1
model.qda.acc = mean(round(model.qda.prediction) == trueLabels)

model.dtree = rpart(formula, data=data, method = 'class')
model.dtree.pred = predict(model.dtree, test.data, type='prob')[,2]
model.dtree.acc = mean(round(model.dtree.pred) == trueLabels)

model.lda.pred$posterior[,2]

#
par(pty = 's')
roc(trueLabels, model.log.pred, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.thres = T, print.auc.y=70, print.auc = TRUE)

plot.roc(trueLabels, model.lda.pred$posterior[,2],percent=TRUE, col="#4daf4a", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=60)

plot.roc(trueLabels, model.qda.pred$posterior[,2],percent=TRUE, col="#cac44d", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=50)

plot.roc(trueLabels, model.dtree.pred,percent=TRUE, col="#b8377e", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=40)

legend("bottomright", legend=c("Logisitic Regression", "LDA", "QDA", "Decision Tree"), col=c("#377eb8", "#4daf4a", "#cac44d", "#b8377e" ), lwd=4)
 #

#par(pty = 's')
par(mfrow = c(2, 2))
roc(trueLabels, model.log.pred, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.thres = T, print.auc.y=70, print.auc = TRUE)
title("Logisitic Regression")

roc(trueLabels, model.lda.pred$posterior[,2], plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#4daf4a", lwd=4, print.thres = T, print.auc.y=70, print.auc = TRUE)
title("LDA")

roc(trueLabels, model.qda.pred$posterior[,2], plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#cac44d", lwd=4, print.thres = T, print.auc.y=70, print.auc = TRUE)
title("QDA")

roc(trueLabels, model.dtree.pred, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#b8377e", lwd=4, print.thres = T, print.auc.y=70, print.auc = TRUE)
title("Decision Tree")

#We will choose ___ as our best model

#good trade off

data.frame(intercept = rep(NA, 10))
#All the ROC of 4 models have very close AUC. QDA has the highest AUC amoung, which makes it the best model. It also matches the best model we concluded 

#the cutoff has the best tradeoff between True Positive Rate and False Positive Rate. We
```


```{r}
model.qda

model.lda.pred$x
```

```{r}
?qda
```

```{r}

x = matrix(c(1,5,64,8,34,52,14,7,85),3,3)

cov(x)


```




