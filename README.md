# Airbnb-Pricing-Optimization
This project aims to develop a smart pricing tool for the hosts of Airbnb, in order to maximize profit. 
Application-specific feature engineering and 
a "**Meta holdout scheme with OOF meta-features**" architecture is deployed in the pricing ML pipeline.
Moreover, we create a clustering (unsupervised) model to help customers get similar recommendations.

## Pricing tool
Main challenge of this project was to develop a pricing tool that can suggest a price (or a price range) to hosts. It is important in
order for this process to be meaningful, to exclude highly overpriced flats and exclude from the
dataset logins of flats with very low occupancy. Underlying idea is that we want to suggest a price based on active and successful 
hosts, not on "ghost-flats" that would add noise to our ML models, in order to optimize hosts profit. 
Established on this idea, we optimize the pricing ML pipeline by application-specific data preprocessing and feature engineering  which among other steps include:

1. **Demand analysis**: Main challenge with the current dataset was to find a way to get the occupancy rate for 
each host or for a market. 
"_Inside Airbnb_", the website that originally gathered and posted the Airbnb data, uses a model which is base on the 
statistics of an average 
stay of the visitors and the probability of leaving a review. They call it the "**San Francisco Model**". Detailed information on 
the "San Francisco Model" is included.
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
Although outliers were trimmed, price distribution was still right skewed.
Further feature engineering includes taking the logarithmic values of the target (price) as it makes a more even 
distribution of the data,
hence showcasing further boost of the model.


### ML architecture
For the ML pipeline of the pricing tool, we create a "**Meta holdout scheme with OOF meta-features**" architecture. 
Such implementation proves to be performing great on this data, since it is computationally and time efficient, 
while providing promising results.
A diagram of the "Meta holdout scheme with OOF meta-features" is presented below followed by the architecture analysis.
<img src=https://github.com/Harry-Kouraklis/Airbnb-Pricing-Optimization/blob/master/Meta%20holdout%20scheme%20with%20OOF%20meta-features.png alt="alt text" width=89% height=89%>

1. We split train data into K folds. Iterating though each fold: retrain 3 diverse models on all folds except the current fold. We then predict for the current fold.

2.  After this step for each object in train_data we will have 3 meta-features (also known as out-of-fold predictions, OOF). Let's call
them meta_train_set

3.  We predict for test data. Let's call these features meta_test_set.

4.  We split meta_train_set into two parts: train_metaA and train_metaB. Fit a meta-model to train_metaA while validating its 
hyperparameters on train_metaB.

When the meta-model is validated, we fit it to train_meta and predict for test_meta.

The three algorithms we will be using are: **Catboost**, **Light GBM** and **Random Forrest**. The **meta-model** is a second **Catboost** Regressor.

The selection of the models is made in a way to provide diversity to the predictions. Underlying idea is that we should search for 
models that "do not make the same mistakes". By having diverse models that make different mistakes, the meta-model is able to "learn"
from these mistakes and provide a better than any of the individual models.

Results showcase a **mean absolute error of $37.66**. Taking into account the high unpredictability of pricing decisions of different hosts and the noisy data, results appear rather satisfying. However, it would probably be be better to suggest a "price range" to hosts than a price.

##  Recommendation tool
For the recommendation tool the concept was to be able to suggest similar flats to customers who are browsing the Airbnb website.
We use **Principal Component Analysis** followed by a **K-Means** unsupervised model. 
PCA is performed for a 95% variance. Analysis includes "Elbow theorem" and "Silhuette scores" plotting 
in order to find the optimum number of clusters. 
Both plots indicate that three is the best choice of number of clusters, achieving a silhuette score of 0.48.

## Future work
Throughout our analysis we saw that even similar flats could have very different prices. 
In our analysis we did not include any unsupervised methods in the regression (ML) pipeline. An interesting concept to work on
could be to cluster hosts following similar pricing strategies (meaning that two flats within the same cluster, 
with the same charasteristics, would have  very similar prices). This way we would be able to denoise data within the same clusters, followed up by seperate ML models for each cluster, in order to boost the ML performance.
