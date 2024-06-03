# Hotel Booking Exploratory Data Analysis
# Install required packages
install.packages("ggplot2")
install.packages("dplyr")
install.packages("scales")
install.packages("tidyverse")
install.packages("readr")
install.packages("zeallot")
install.packages("countrycode")
install.packages("ISLR")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("rattle")
install.packages("GoodmanKruskal")
install.packages("arm")
install.packages("randomForest")
install.packages("naniar")
install.packages("lubridate")
install.packages("e1071")
install.packages("randomForest")
install.packages("pROC")
install.packages("corrplot")
install.packages("fastDummies")
install.packages("gridExtra")
install.packages("GGally")
install.packages("knitr")
install.packages("akmedoids")
install.packages("FSinR")
install.packages("party")
install.packages("reshape2")
install.packages("useful")
install.packages("NbClust")
install.packages("stats")
install.packages("cluster")
install.packages("purrr")
install.packages("Rpdb")
install.packages("xgboost")
install.packages("factoextra")
install.packages("ROCR")
install.packages("klaR")



# Load all the required library
library(ggplot2)
library(dplyr)
library(scales)
library(tidyverse)
library(readr)
library(zeallot)
library(countrycode)
library(ISLR)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(GoodmanKruskal)
library(arm)
library(randomForest)
library(naniar)  # handling missing data
library(lubridate) # convert Date
library(corrplot)
library(e1071)
library(randomForest)
library(pROC) 
library(fastDummies)
library(gridExtra)
library(GGally)
library(knitr)
library(akmedoids)
library(FSinR)
library(party)
library(reshape2)
library(useful)
library(NbClust)
library(stats)
library(cluster)
library(purrr)
library(Rpdb)
library(xgboost)
library(factoextra)
library(ROCR)
library(klaR)

# Import dataset
location <- 'hotel_bookings.csv'
data <- read.csv(location)
View(data)

# summarize the data
summary(data)
str(data)

##############################
#       DATA CLEANING        #
##############################

##### NULL and NA values evaluation

# find NA values
colSums(is.na(data))

# Children columns has 4 NA values, delete the rows that contain NA values:
data = na.omit(data)
# CHeck for NA again, no more NA values
colSums(is.na(data))

# Check if dataset contain "NA" value
print(miss_scan_count(data = data, search = list("NA")), n=ncol(data))

# 3 values in Countries was valued as NA, delete them
data <- data[!grepl("NA", data$country),]
# Check if delete is scuccess
print(miss_scan_count(data = data, search = list("NA")), n=ncol(data))

# Count number of null
print(miss_scan_count(data = data, search = list("NULL")), n=ncol(data))

# Keep null in agent and company, as they just mean the customer has not book through any agent or company
# 418 NULL values in Country, delete them
data <- data[!grepl("NULL", data$country),]

# Check if delete is scuccess
print(miss_scan_count(data = data, search = list("NULL")), n=ncol(data))


# Count number of Undefined
print(miss_scan_count(data = data, search = list("Undefined")), n=ncol(data))

# 1165 Undefined values in meal and 1 in distribution channel, delete them
data <- data[!grepl("Undefined", data$meal),]
data <- data[!grepl("Undefined", data$distribution_channel),]
print(miss_scan_count(data = data, search = list("Undefined")), n=ncol(data))


########## Logical error elimination
## Eleminate entry with no number of customer
error <-data[data$adults == 0 & data$children == 0 & data$babies == 0,]
error

data <- anti_join(data,error)
View(data)

## Eleminate entry where avarge daily rate <0
data <- subset(data, data$adr >= 0)
data
## Eleminate entry where avarge daily is a outlier
data <- subset(data, data$adr < 5400.0)
data

##############################
#       Data Transform       #
##############################
#Create Country name column
data$country[which(data$country == "CN")] <- "CHN"
data$country[which(data$country == "TMP")] <- "TLS"

data$country_name <- countrycode(data$country, 
                                 origin = "iso3c",
                                 destination = "country.name")

#Create continent column
data$continent <- countrycode(data$country, 
                              origin = "iso3c",
                              destination = "continent")
data = na.omit(data)
# Change the value of Continent if the booking is from domestic
data$continent[which(data$country_name == "Portugal")] <- "Europe_Domestic"

##############################
#            EDA             #
##############################

# Visualize the distribution
ggplot(data = data,
       aes(
         x = hotel,
         y = prop.table(after_stat(count)),
         fill = factor(is_canceled),
         label = scales::percent(prop.table(stat(count)))
       )) +
  geom_bar(width = 0.75, position = position_dodge()) +
  geom_text(
    stat = "count",
    position = position_dodge(0.8),
    vjust = -0.5,
    size = 5
  ) + 
  coord_flip()+
  theme_minimal()+
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Cancellation Status by Hotel Type",
       x = "Hotel Type",
       y = "Count") +
  scale_fill_discrete(
    name = "Booking Status",
    breaks = c("1", "0"),
    labels = c("Cancelled", "Not Cancelled")
  )

## City hotel has more than 65% of customers, 

# Histogram on Lead_time
ggplot(data = data, aes(lead_time)) + 
  scale_fill_brewer(palette = "Spectral") +
  geom_histogram(aes(fill=factor(is_canceled)), 
                   binwidth = 30, 
                   col="black", 
                   size=.1) + 
  stat_bin(binwidth=30, geom="text", colour="black", size=3.5,
           aes(label=..count.., group=factor(is_canceled), y=0.8*(..count..))) +
  labs(title="Histogram of Lead_time by Booking Status", y = "Count")+
  scale_fill_discrete(
    name = "Booking Status",
    breaks = c("1", "0"),
    labels = c("Cancelled", "Not Cancelled")
  ) +theme_minimal()+
  coord_cartesian(xlim = c(0, 300))+
  scale_x_continuous(breaks=seq(0,300, 30))


# note this data use the booking confirmation and not the check ins, so this graph shows the
# booking made for particular month and not the confirmed check ins.
data$arrival_date_month <-
  factor(data$arrival_date_month, levels = month.name)

ggplot(data, aes(arrival_date_month, fill = factor(is_canceled))) +
  geom_bar() + geom_text(stat = "count", aes(label = ..count..), hjust = 1) +
  coord_flip() + scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    label = c("Not Cancelled", "Cancelled")
  ) +
  labs(title = "Booking Status by Month",
       x = "Month",
       y = "Count") 


#where are the people coming from
data_1 <- data[,]
# Subset the data to include the countries which has more than 1500 reservation request
# otherwise including all the country with few or occassional request to avoid the graph
# from being clumsy
sub_hotel <- data_1 %>% 
  group_by(country) %>% 
  filter(n() > 1500)

# Visualize the Travellor by Country.
sub_hotel$county_name <- countrycode(sub_hotel$country, 
                                     origin = "iso3c",
                                     destination = "country.name")

# Booking Status by Country
ggplot(sub_hotel, aes(county_name, fill =factor(is_canceled))) + 
  geom_bar(stat = "count", position = position_dodge()) + 
  labs(title = "Booking Status by Country",
       x = "Country",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_blank()) + 
  scale_fill_discrete(
    name = "Booking Status",
    breaks = c("1", "0"),
    labels = c("Not Cancelled", "Cancelled"))


# Does the hotel charged differently for different market_segment
ggplot(sub_hotel, aes(x = market_segment, y = adr, fill = hotel)) + 
  geom_boxplot(position = position_dodge()) + 
  labs(title = "Price Charged by Market Segment",
       subtitle = "for Customer Type",
       x = "Market Segment",
       y = "Price per night(in Euro)")

##########################
#   Data Transformation  #
##########################
data_num = data[ , purrr::map_lgl(data, is.numeric)]
drop <- c("is_canceled")
data_num  = data_num[,!(names(data_num) %in% drop)]
View(data_num)
# scale numeric data
data_num_scale <- as.data.frame(scale(data_num))
View(data_num_scale)


data$arrival_date_quarter <- NA # need to initialize variable
data$arrival_date_quarter[data$arrival_date_month == "November" | data$arrival_date_month == "December" | data$arrival_date_month == "January"] = "Winter"
data$arrival_date_quarter[data$arrival_date_month == "February" | data$arrival_date_month == "March" | data$arrival_date_month == "April"] = "Spring"
data$arrival_date_quarter[data$arrival_date_month == "May" | data$arrival_date_month == "June" | data$arrival_date_month == "July"] = "Summer"
data$arrival_date_quarter[data$arrival_date_month == "August" | data$arrival_date_month == "September" | data$arrival_date_month == "October"] = "Fall"

# df <- data.frame(year = as.character(format(data$reservation_status_date, format = "%Y")),
#                  month = as.character(format(data$reservation_status_date, format = "%m")),
#                  day = as.character(format(data$reservation_status_date, format = "%d")))
# View(df)
# 
# # data$reservation_status_day = df$day
# data$reservation_status_month = df$month
# data$reservation_status_year = df$year

data$booking_with_agent <- NA
data$booking_with_agent[data$agent == "NULL"] = "No"
data$booking_with_agent[data$agent != "NULL"] = "Yes"

data$booking_with_company <- NA
data$booking_with_company[data$company == "NULL"] = "No"
data$booking_with_company[data$company != "NULL"] = "Yes"

data$room_expectation_match <- NA # need to initialize variable
data$room_expectation_match[data$reserved_room_type == data$assigned_room_type] = "Yes"
data$room_expectation_match[data$reserved_room_type != data$assigned_room_type] = "No"

data_cat = data[ , purrr::map_lgl(data, is.character)]
data_cat$is_canceled = data$is_canceled
View(data_cat)
data_cat <- data_cat[,!names(data_cat) %in% c("agent", "company","assigned_room_type","country","country_name", "reservation_status", "reservation_status_date")]

View(data_cat)


## Get dummies variable
data_cat_dummy = data_cat[,]
View(data_cat_dummy)
# Creating dummy columns
data_cat_dummy <- dummy_cols(data_cat_dummy,select_columns=colnames(data_cat_dummy),
                             remove_selected_columns = TRUE, remove_first_dummy = TRUE)

names(data_cat_dummy)<-make.names(names(data_cat_dummy),unique = TRUE)

# View a summary of the data
summary(data_cat_dummy)

data_cat_dummy = data_cat_dummy[,]
View(data_cat_dummy)

data_cat_num<- as.data.frame(bind_cols(data_num_scale, data_cat_dummy)) 

names(data_cat_num)<-make.names(names(data_cat_num),unique = TRUE)
View(data_cat_num)

#########################
#    Data Selection     #
#########################

# Select by correlation


# creating correlation matrix
corr_mat <- round(cor(data_num_scale),2)

# reduce the size of correlation matrix
melted_corr_mat <- melt(corr_mat)
head(melted_corr_mat)

# plotting the correlation heatmap
ggplot(data = melted_corr_mat, aes(x=Var1, y=Var2,
                                   fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),
            color = "black", size = 4)


# Select by random forest
cf1 <- cforest(is_canceled_1 ~ . , data= data_cat_num, control=cforest_unbiased(mtry=2,ntree=3)) # fit the random forest
varimp(cf1)

imp <- as.data.frame(varImp(cf1))
imp <- data.frame(overall = imp$Overall,
                  names   = rownames(imp))
imp_rf <- imp[order(imp$overall,decreasing = F),]
imp_rf

## Remove feature with 0.00 important score
imp_rf_drop <- imp[imp$overall==0,]
imp_rf_drop
#Select features with important score not equal to zero
data_cat_num  = data_cat_num[,!(names(data_cat_num) %in% imp_rf_drop$names)]
View(data_cat_num)

data_cat_num[sapply(data_cat_num, is.integer)] <-
  lapply(data_cat_num[sapply(data_cat_num, is.integer)], as.factor)
str(data_cat_num)

summary(data_cat_num)

data_cat_num <- na.omit(data_cat_num) 

#########################
#      Clustering       #
#########################


data_clust_dummy = data_cat[,]
data_clust_dummy <- dummy_cols(data_clust_dummy,select_columns=colnames(data_clust_dummy),
                             remove_selected_columns = TRUE)
names(data_clust_dummy)<-make.names(names(data_clust_dummy),unique = TRUE)
data_clust_num <- as.data.frame(bind_cols(data_num,data_clust_dummy)) 
data_clust_num_scale <- as.data.frame(bind_cols(data_num_scale,data_clust_dummy)) 
names(data_clust_num)<-make.names(names(data_clust_num),unique = TRUE)
View(data_clust_num)

set.seed(42)
data_cat_num_frac <- data_cat_num 
data_clust_sample <- sample_frac(data_clust_dummy, 0.02)
View(data_clust_sample)


fviz_nbclust(data_clust_sample, FUN = hcut, method = "wss")

# methods to assess
m <- c( "average", "single", "complete", "ward")
names(m) <- c( "average", "single", "complete", "ward")

# function to compute coefficient
ac <- function(x) {
  agnes(data_clust_sample, method = x)$ac
}

map_dbl(m, ac)

# Dissimilarity matrix
d <- dist(data_clust_sample, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "ward.D2" )
# Plot the obtained dendrogram
plot(hc1)

set.seed(42)
NbClust(data_clust_sample, distance = "euclidean", 
        min.nc=2, max.nc=10, method = "ward.D2", 
        index = "kl")

set.seed(0)
data_k4 <- kmeans(data_clust_dummy, centers=4)
set.seed(1234)
data_clust_k4 <- kmeans(data_clust_sample, centers=4)

plot(data_clust_k4, data=data_clust_sample)

# Mean values of each cluster
set.seed(42)
aggregate(data_clust_sample, by=list(data_clust_k4$cluster), mean)
set.seed(0)
aggregate(data_clust_num, by=list(data_k4$cluster), mean)

data$cluster = data_k4$cluster

df <- data.frame(data %>%
  group_by(cluster, continent, market_segment) %>%
  summarise(total_adr = sum(adr),
            mean_adr = mean(adr),
            sd_adr = sd(adr),
            total_lead_time = sum(lead_time),
            mean_lead_time = mean(lead_time),
            sd_lead_time = sd(lead_time),
            total_is_canceled = sum(is_canceled),
            mean_is_canceled = mean(is_canceled),
            sd_is_canceled = sd(is_canceled),
            count = n(),
            .groups = 'drop')
)
view(df)
#########################
#     Classification    #
#########################
seed = 42
############Data Spliting ########
set.seed(seed)
intrain <- createDataPartition(y = data_cat_num$is_canceled_1,
                               p = 0.7,
                               list = FALSE)

training <-data_cat_num[intrain,]
testing <- data_cat_num[-intrain,]



########### Fit the models on the training set ##########
trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "grid")

# K-nearest neighbour
set.seed(seed)
knn_model <- train(
  is_canceled_1 ~ ., data = training, method = "knn",
  trControl = trControl,
  preProcess = c("center","scale"),
  tuneLength = 3
)

#Decision tree

# Fit the model on the training set
set.seed(seed)
model_dc <- train(
  is_canceled_1 ~ ., data = training, method = "rpart",
  trControl = trControl,
  tuneLength = 5
)

#XGBoost
set.seed(seed)
xgb_model <- train(
  is_canceled_1 ~ ., data = training, method = "xgbTree",
  trControl = trControl
)

#Random Forest


set.seed(seed)
rf_model <- train(is_canceled_1 ~ ., data = training,
                  method = 'ranger',
                  trControl = trControl)


##### Plot model ########

# KNN: Accuracy vs different values of k
plot(knn_model)

# Decision Tree: Accuracy vs different values of cp (complexity parameter)
plot(model_dc)

# XGboost
plot(xgb_model)
# Randomforest
plot(rf_model)


######### Print the best tuning ########### 
#KNN: parameter k that maximizes model accuracy
knn_model$bestTune
model_dc$bestTune
xgb_model$bestTune
rf_model$bestTune

######### feature importance #############
varImp(knn_model)
plot(varImp(knn_model))
varImp(model_dc)
plot(varImp(model_dc))
varImp(xgb_model)
plot(varImp(xgb_model))
rf_model_imp = data.frame(rf_model_imp)
rf_model_imp[order(rf_model_imp$X1, decreasing = FALSE),]


########### Model performance ############
results <- resamples(list(KNN=knn_model, DC=model_dc, XGB =xgb_model, RF= rf_model))
summary(results)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales) 

densityplot(results, scales=scales, pch = "|")
dotplot(results, scales=scales)



########### Prediction - Validation ############
#KNN prediction:

knn_model.pred <- knn_model%>% predict(testing)

confusionMatrix(knn_model.pred, factor(testing$is_canceled_1), positive = "1")


# KNN: ROC curve
knn_valid_pred <- predict(knn_model,testing, type = 'prob')[,2]
knn.roc <- roc(testing$is_canceled_1, knn_valid_pred)
print(knn.roc)
plot(knn.roc, print.auc = TRUE, print.thres = "best")
knn_auc <- auc(testing$is_canceled_1, knn.roc)
knn_auc

# Decision tree train model with best tune
mod_tree <- rpart(is_canceled_1 ~ ., data = training, method = "class", cp = 0.008600469)
# Plot the trees
rpart.plot(mod_tree)

# Decison Tree Prediction:
tree.pred <- predict(mod_tree, testing, type = "class")

confusionMatrix(tree.pred, factor(testing$is_canceled_1), positive = "1")

# Decision Tree: ROC curve
tree.preds <- predict(mod_tree, testing, type = "prob")[,2]
tree.roc <- roc(testing$is_canceled_1, tree.preds)
print(tree.roc)
plot(tree.roc, print.auc = TRUE, print.thres = "best")
dc_auc <- auc(testing$is_canceled_1, tree.preds)


#XGB predcition
xgb_pred <- predict(xgb_model, testing) 
confusionMatrix(xgb_pred, factor(testing$is_canceled_1), positive = "1")


# XGB : ROC curve
xgb_preds <- predict(xgb_model, testing, type = "prob")[,2]
xgb.roc <- roc(testing$is_canceled_1, xgb_preds)
print(xgb.roc)
plot(xgb.roc,print.auc = TRUE, print.thres = "best")
xgb_auc <- auc(testing$is_canceled_1, xgb_preds)


# Random forest: Prediction
rf_model_pred <- predict(rf_model, testing)
confusionMatrix(rf_model_pred, factor(testing$is_canceled_1), positive = "1")

# Random Forest: ROC curve
rf_model_preds <- predict(rf_model, testing, type = "prob")[,2]
rf.roc <- roc(testing$is_canceled_1, rf_model_preds)
print(rf.roc)
plot(rf.roc, print.auc = TRUE, print.thres = "best")
rf_auc <- auc(testing$is_canceled_1, rf_model_preds)


######### Prediction evaluation: #############

mean_rf = mean(rf_model_pred == testing$is_canceled_1)
sd_rf = sd(rf_model_pred == testing$is_canceled_1)

mean_knn =mean(knn_model.pred == testing$is_canceled_1)
sd_knn = sd(knn_model.pred == testing$is_canceled_1)

mean_xgb =mean(xbg_pred == testing$is_canceled_1)
sd_xgb = sd(xbg_pred == testing$is_canceled_1)

mean_dc =mean(tree.pred == testing$is_canceled_1)
sd_dc = sd(tree.pred == testing$is_canceled_1)
model_name = c("K-nearest Neighbour", "Decision Tree", "XGBoost", "Random Forest")
model_mean = c(mean_knn, mean_dc, mean_xgb, mean_rf)
model_sd = c(sd_knn, sd_dc, sd_xgb, sd_rf)
model_auc = c(knn_auc, dc_auc, xgb_auc, rf_auc)

model_df<- data.frame(model_name, model_mean, model_sd, model_auc)
model_df








