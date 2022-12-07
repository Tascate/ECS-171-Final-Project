# ECS-171-Final-Project

## Introduction
For this project we have decided to choose to predict whether someone buys a product or not when online shopping. It is cool because a lot of people shop online today and that using other information we can come to a conclusion 
whether the product will be bought or not. Knowing this information would also let websites to advertise the product to people who might buy it which would incentivize them more to buy the product rather than advertising to 
people who would not even look at the product. Having a good predictive mode is important as it improves the accuracy of the prediction and would be easier to improve on new data sets. Being more accurate also means being more 
reliable when using them.  

## Figure

![alt text](https://github.com/Tascate/ECS-171-Final-Project/blob/main/plots.png?raw=true)

## Methods
## Data Exploration

The [online shopper intention dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#) focuses on shopper website activity and intention to purchase. The dataset indicates whether the shopper resulted in revenue for the website or not through true or false. In total there are 12,330 observations within the dataset. Of which, 1,908 resulted in a purchase. 

The dataset contains specific data about the user's website surfing such as viewing webpages relating to administration, informational, or products and how the time spent on each web page category. Data from Google Analytics is also included which provide the average "bounce rate", average "exit rate" and average "page rate" for a specific web page. There is also additional data about the user such as operating system, browser, region, traffic type, new or returning vistior, and date specifics. 

For many data entries, administrative and informational web pages is entered as 0 indicating that the user has not viewed those web pages at all.

## Data Analysis

Data in the table would have to be normalized as each feature has wildy different ranges. This especially goes true for web page durations as it can range from 0 to the thousands.

A majority of the entries that resulted in a shopping purchase also had PageValues that were non-zero. PageValues being the average number of web pages that a user has visited before completing an e-commerce transaction for that product.

The data includes information about the user as well as statistics from Google Analytics detailing the average user profile who visited that website. The model would be relating looking at a user's profile along with the average user profile to determine if it is likely that a user would shop.

## Data Preprocessing

The data within the table would have to be normalized since number ranges for each feature vary widely. The final column representing purchase would become the y in our model while the rest of the features would become the x.

The administrivate and informational columns along with their respective duration columns can be dropped from the dataset. A majority of the entires have zero/missing values for them and may not be helpful for the model as a result.

Additionaly, the PageValues column can also be dropped from the dataset. A majority of the entries have zero values for them and usually entries that resulted in shopping have a non-zero PageValue. This feature may not be helpful for the model because of this.

```
data_z[column] = MinMaxScaler().fit_transform(np.array(data_z[column]).reshape(-1,1))
```

## First Model Building

For the first model, we select SVM model because the dataset has both categorical and numerical attributes. 

We first drop the useless columns mentioned in Data Preprocessing and we find the type of each column so that we can encode those categorical attributes. Then we find that there is one column which needs to be normalized.

We normalized the column with large scale and then we use one-hot-encoding to turn Month and VisitorType into numerics so that we can apply SVM model with rbf kernel.

We split the data into 80:20 training and testing subsets and train the model with rbf kernel.

We get the precision and classification report to define the performance of our training and testing.

```
onehot=OneHotEncoder()
transformed_month=onehot.fit_transform(data[["Month"]])
data[onehot.categories_[0]] = transformed_month.toarray()

transformed_vistype=onehot.fit_transform(data[["VisitorType"]])
data[onehot.categories_[0]] = transformed_vistype.toarray()

data=data.drop(["Month","VisitorType"],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
```

## Results
For checking our results we used accuracy score and classification to print out the report below. We receive that the accuracy score is 86% which is good accuracy for the model. 
```
              precision    recall  f1-score   support

       False       0.86      1.00      0.92      2123
        True       0.00      0.00      0.00       343

    accuracy                           0.86      2466
   macro avg       0.43      0.50      0.46      2466
weighted avg       0.74      0.86      0.80      2466
```

## Discussion
After choosing the dataset we had to decide which model to use. We were going to use logistic regression as it was simple to work on but later on switched to SVM as the dataset had numerical and categorical which was the right idea. 

After choosing the model we began preprocessing the data by normalizing it and deleting unnecessary columns. This is one area in which we could have done better by choosing a better dataset to train on which did not have a lot values which were 0 and could have developed a model that takes more attributes into consideration. 

Then we onehot encode the categorical variable so we can apply the rbf kernel to it and then split the data into testing and training with the split of 80:20. Then we print out the accuracy score and classification report to see how well our model can predict whether a person makes a purchase or not. 

## Conclusion 
Doing this project was a fun way to learn what we could do with machine learning and how it can be useful in life. What we think we could have done differently is to choose a different dataset with more variables that did not have mostly zeroes and make the model even more accurate using most of the variable.

## Collaboration Section
Everyone gave feedback to each other and helped the each other out when one wasn't sure what to do. 

Troy Insixiengmay: Did the abstract and chose what the project should be about and did the data exploration and analysis.

Yuhan Pu: Did all the data preprocessing and built the model and did most of the code.

Govind Alagappan: Did all of the write up for the final submission.