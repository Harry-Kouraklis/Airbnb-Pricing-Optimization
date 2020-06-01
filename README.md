# Airbnb-Pricing-Optimization
This project aims to develop a smart pricing tool for the hosts of Airbnb, in order to maximize profit. 
Application-specific feature engineering and 
a "Meta holdout scheme with OOF meta-features" architecture took place in the pricing ML pipeline.
Moreover, we create a clustering (unsupervised) model to help customers get similar recommendations.

For the **pricing tool**, our aim is to create an algorithm model that does not stand as a predictor of flat prices 
but rather as a pricing tool for profit optimization. 
Underlying idea is the fact that we can optimize the pricing ML pipeline by specific-purpose data preprocessing and feature engineering. 
Feature engineering among others include:

1. **Demand analysis**: Main challenge with the current dataset was to find a way to getting the occupancy rate for 
each host or for a market. 
"Inside Airbnb", the website that originally gathered and posted the Airbnb data, uses a model which is base on the 
statistics of an average 
stay of the visitors and the probability of leaving a review.They call it the "**San Francisco Model**". Detailed information on the 
"San Francisco Model" is included.
Through proper demand analysis we are able to locate and exclude ghost-flats (flats with very low occupancy rate 
through-out the year) from our dataset, leading to a significant boost of the model's performace. 
Model not only performs better but also suggests prices that will actually benefit the host. 
2. **Distance from NY tourist attractions**. 
We enhance the features of our dataset by including the flats distance from famous NY tourist attractions as we believe that 
there is indeed an inverse relation between the mentioned distance and the price.
3. **Under-represented categorical data**. We examine under-represented categorical data. More specifically, 
some of the categories of column "neighbourhood" consist 
of areas that have insufficient amount of indexes. It is very likely that our models "memorize" these indexes without any 
generalization value. For such indexes, we change the name of the area to "other" to bypass this issue.
4. **Price trasformation**:
Although outliers were trimmed, price values distribution was still right skewed.
Further feature engineering includes taking the logarithmic values of the target (price) as it makes a more even 
distribution of the data,
hence showcasing further boost of the model.


**ML architecture**: For the ML pipeline of the pricing tool, we create a "**Meta holdout scheme with OOF meta-features**" architecture. 
Such implementation proves to be performing great on this data, since it is computationally and time efficient, 
while providing promising results.

<img src=https://github.com/Harry-Kouraklis/Airbnb-Pricing-Optimization/blob/master/Meta%20holdout%20scheme%20with%20OOF%20meta-features.png alt="alt text" width=90% height=90%>

For the recommendation tool, we use Principal Component Analysis followed by a K-Means unsupervised model. 
PCA is performed for a 95% variance. Analysis includes "Elbow theorem" and "Silhuette scores" plotting 
in order to find the optimum number of clusters. 
Both plots indicate that three is the best choice of number of clusters, achieving a silhuette score of 0.48.
