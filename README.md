# ECS-171-Final-Project

## Data Exploration

The dataset focuses on shopper website activity and intention to purchase. The dataset indicates whether the shopper resulted in revenue for the website or not through true or false. In total there are 12,330 observations within the dataset. Of which, 1,908 resulted in a purchase. 

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