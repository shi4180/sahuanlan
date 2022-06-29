library(readxl)
library(glmnet)
library(pROC)
databziltrain <- read.csv("mrmrtraindata.csv")
fit = glmnet(databziltrain[2:101], databziltrain$Class, family="binomial", nlambda=50, alpha=1)
fit
fit$lambda
a <- databziltrain[2:101]
b = as.matrix(a)
opar <- par(no.readonly=TRUE)
par(pty="s", mai=c(1,0,0.5,0))
plot(fit)
cvfit = cv.glmnet(b, databziltrain$Class, family = "binomial", type.measure = "auc")
plot(cvfit)


cvfit



par(opar)
model_lasso <- glmnet(x=b, databziltrain$Class, alpha = 1, lambda=cvfit$lambda.1se)
lasso.prob <- predict(cvfit, newx=b , type="response",s=c(cvfit$lambda.1se) )
assess.glmnet(lasso.prob, newy = databziltrain$Class, family = "binomial")
lasso.prob
model_lasso
print(lasso.prob)

databziltest <- read.csv("mrmrtestdata.csv")
a2 <- databziltest[2:101]
b2 = as.matrix(a2)
lasso.prob2 <- predict(cvfit, newx=b2 , type="response",s=c(cvfit$lambda.1se) )
assess.glmnet(lasso.prob2, newy = databziltest$Class, family = "binomial")
lasso.prob2

databzval <- read_excel("mrmrvaldata1.xlsx")
a3 <- databzval[2:101]
b3 = as.matrix(a3)
lasso.prob3 <- predict(cvfit, newx=b3 , type="response",s=c(cvfit$lambda.1se) )
assess.glmnet(lasso.prob3, newy = databzval$CLNM, family = "binomial")
lasso.prob3




yuce <- data.frame(lasso.prob)
write.csv(yuce,file = "yuce.csv")
xishu <- coef(cvfit,s=cvfit$lambda.1se)
xishu
yuce <- read.csv("yuce.csv")
roc1 <- roc(yuce$Class,yuce$s1)
roc1
nrow(xishu)
roc1$sensitivities
ci(roc1)
yuce2 <- data.frame(lasso.prob2)
write.csv(yuce2,file = "yuce2.csv")
yuce2 <- read.csv("yuce2.csv")
roc2 <- roc(yuce2$Class,yuce2$s1)
roc2
ci(roc2)

yuce3 <- data.frame(lasso.prob3)
write.csv(yuce3,file = "yuce3.csv")
yuce3 <- read_excel("yuce3.xlsx")
roc3 <- roc(yuce3$CLNM,yuce3$s1)
roc3
ci(roc3)


