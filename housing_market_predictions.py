"""
Boston Housing Market Project

Machine Learning NanoDegree - Udacity

Pavlos Sakoglou 

"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

import visuals as vs

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

# Comment out the following line to see the prices
#print(prices)

#######################################################################

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("\nStatistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))


''' 
Question 1:
    Using your intuition, for each of the three features above, do you think 
    that an increase in the value of that feature would lead to an increase in 
    the value of 'MEDV' or a decrease in the value of 'MEDV'? Justify your 
    answer for each.
    
Answer:
    Let's examine things carefully:
        
    Since these features reflect the quality and level of life in these neighborhoods,
    it is natural to assume that possible increases or decreases in value will determine
    the demand of the market over the houses, which will reflect in higher or lower
    prices. 
        
    An increase of the values of LSTAT feature, which represents the percentage 
    of lower class people living in a neighborhood, won't be approaching a 
    "better neighborhood" value, and as a result an increase of this feature more 
    likely will decrease the value of the house prices, since wealthy people, who 
    are willing to pay more, want to live in wealthier areas. 
      
    An increase in RM i.e. if the average number of rooms per house increase, 
    which means if the size of the houses is getting bigger, will imply that the 
    neighborhood is wealthier, since large houses are found in wealthier areas. 
    That means, that the prices will also go up, as the size of the houses is 
    going up. 
    
    For the PTRATIO, if its values increase, it means that there are many students
    per teacher, which means that on average the level of education is decreased. 
    Prestigious schools are famous for having small size classes so that the students
    will get more individual attention by the teacher and thus comprehend the
    material better. Non prestigious schools, like most public schools, are famous 
    (or infamous) for the opposite: large size classrooms, prioritizing quantity 
    over quality. Thus, increased PTRATIO values means the level of education 
    is decreasing, which means that wealthy potential buyers won't be much interested
    in such neighborhoods cause they want their kids to go to good schools, 
    which in turn will decrease the demand, and as a result the prices of the 
    houses in these neighborhoods. 

    In summary:
        
        As RM increases -> MEDV increases
        As LSTAT increases -> MEDV decreases
        As PTRATION increases -> MEDV decreases
'''

# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("\nModel has a coefficient of determination, R^2, of {:.3f}.".format(score))

''' 
Question 2:
    Would you consider this model to have successfully captured the variation of the target variable?
    Why or why not?

Answer:
    Yes. The output of the above metric given the data above is 0.923, which is in a  
    neighborhood of 1 with radius = 0.077. Sounds pretty good score given that
    a score of 1 perfectly predicts the target variable. 
    
'''

# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=20, random_state=True)

# Success
print("Training and testing split was successful.")


'''
Question 3:
    What is the benefit to splitting a dataset into some ratio of training and 
    testing subsets for a learning algorithm?
    
Answer:
    It is useful to split the data into training and testing subsets because that
    way we can use some of them (most of them usually) to train the model, and then 
    use the rest of the data to test the model. That enables us to extract information
    about the model, how well it performs, if it overfits or underfits the data etc. 
    
    If we don't split the data and use all of them for training, then we won't 
    have any way of knowing how the model might perform in real data. This is a
    good technique to help us evaluate our model choices. 
    
    (This is from lesson 6: testing models)

'''

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)

'''
Question 4:
    Choose one of the graphs above and state the maximum depth for the model.
    What happens to the score of the training curve as more training points are added? 
    What about the testing curve?
    Would having more training points benefit the model?
    
Answer:
    I am choosing the second curve with max_depth = 3, which is highly biased. Deeper-depth
    trees give better accuracy on the training set, and less deep-depth trees give
    better accuracy on the testing set. Moreover, deep trees let the model overfit
    the data and create a closely fit decision boundary which often does not
    correspond to the actual boundary.
    
    With:
        max_depth: 3
        Testing score: blue curve
        Training score: red curve
    
    We have the following conclusions:
        1) The score of training curve decreases the more we add training points
            and at some point stabilizes -- it's still high
        2) The score of testing curve increases the more we add training points
           and at some point stabilizes -- it's still high
          
    The more training points we have, the lesser the training score will be and the 
    higher the testing score will be until they converge, which implies a highly 
    biased model. 

'''

vs.ModelComplexity(X_train, y_train)

'''
Question 5:
    When the model is trained with a maximum depth of 1, does the model suffer 
    from high bias or from high variance?
    How about when the model is trained with a maximum depth of 10? 
    What visual cues in the graph justify your conclusions?

Answer:
    When the model is trained with a max depth of 1, then it suffers from high bias
    When the model is trained with a max depth of 10, then it suffers from high variance.
    The above statements are obvious from the curves, by observing how they behave
    as # of depth -> inf. 
    The more the # of depth grows, the training score is pretty high, while the 
    validation score decreases. That means that as the # of depth increases, the more
    our model "memorizes" the data instead of learning from them, which makes the validation
    curve to decrease, cause the model won't validate as good with high bias.


Question 6: 
    Which maximum depth do you think results in a model that best generalizes to unseen data?
    What intuition lead you to this answer?

Answer:
    Ideally we would want our model to have a good trade off between variance and bias.
    That is, we don't want our model to neither overfit (high variance) nor underfit
    (high bias). To achieve that, I believe that the max_depth should be 3 or 4.
    
    I arrived to this conclusion by looking at the graphs. The first graph with the 
    4 plots shows us that a good fit and trade off is achieved with depth 3 versus
    the other graphs with increased depth. Then the second graph with the validation 
    curve shows that the validation score is in an acceptable region between max depth
    3 and 4.  

'''

'''
Question 7:
    What is the grid search technique?
    How it can be applied to optimize a learning algorithm?
    
Answer:
    The grid search technique is used when we have a set of models to choose from, 
    which differ from each other in their parameter values that lie on a gird.
    We then train each model and evaluate them using cross-validation and select 
    the one that performed best. 
    
    Further research in wikipedia (https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)):
        
        Hyperparameter optimization is the problem of choosing a set of optimal 
        hyperparameters for a learning algorithm.
        
        The traditional way of performing hyperparameter optimization is grid search,
        which is simply an exhaustive searching through a manually specified subset 
        of the hyperparameter space of a learning algorithm. 
        
        A grid search algorithm must be guided by some performance metric, 
        typically measured by cross-validation on the training set or evaluation 
        on a held-out validation set.
    
    

Question 8:
    What is the k-fold cross-validation training technique?
    What benefit does this technique provide for grid search when optimizing a model?

Answer:
    
    K-fold validation is when we split our data into K subsets and use each subset
    separately as testing data, and the rest points as training data. We then train 
    our model K different times, one for each subset, and then we average the results
    to get the final model. 
    
    To do grid search, iterate through a wide range of grid combinations. 
    Then, choose the set of parameters for which k-folds reports the lowest error.            
    
'''


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
  
    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state = 0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {"max_depth" : [1,2,3,4,5,6,7,8,9,10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

'''
Question 9:
    What maximum depth does the optimal model have? 
    How does this result compare to your guess in Question 6?
    
Answer:
    "Parameter 'max_depth' is 4 for the optimal model."
    
    In Question 6, by looking at the plots I assumed a max depth of 3 or 4. 
    Apparently there is another correlation here that I missed. If you look at 
    the second graph, the validation reaches a maximum value at depth 4. If this is
    true, then our job becomes easier to quickly get an idea of how the max depth would
    look like for our model.    

'''

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print ("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))


'''
Question 10:
    What price would you recommend each client sell his/her home at?
    Do these prices seem reasonable given the values for the respective features?
    
Answer:
    As per the model's output:
        Client 1 should sell at: $408,800.00
        Client 2 should sell at: $231,675.00
        Client 3 should sell at: $931,350.00
    
    Client 1 has a 5-bedroom house (which is large enough), a decent poverty level
    at 17%, and a good student/teacher ratio. We can't draw any conclusions by solely
    examining this client. So let's look at the other two.
    
    Client 2 has a 4-room house (which is relatively smaller), higher poverty rate, 
    almost double in fact, and a border-line average student/teacher ratio. Without
    looking at the predicted price, We should expect it to be lower since all the 
    characteristics are worse than Client 1's characteristics. 
    Indeed the price makes sense to be smaller at $231,675
    
    Client 3 has a significantly larger house (8 rooms is a palace!), almost no lower
    class population (3% poverty), and a great student/teacher ratio, much better
    than both of the other clients. Thus, we expect the price to be much higher. 
    Indeed the predicted price is $931,350, which is more than double from Client 1.
    I would personally expect it to be around double, but I guess the poverty rate 
    which itself is almost 6 times less than Client 1's neighborhood makes the difference, 
    along with the extra 3 bedrooms. 
    
'''

vs.PredictTrials(features, prices, fit_model, client_data)

'''
Question 11:
    In a few sentences, discuss whether the constructed model should or 
    should not be used in a real-world setting.
    
    In particular:
        How relevant today is data that was collected from 1978? How important is inflation?
        
        Are the features present in the data sufficient to describe a home? 
        Do you think factors like quality of apppliances in the home, square 
        feet of the plot area, presence of pool or not etc should factor in?
        
        Is the model robust enough to make consistent predictions?
        
        Would data collected in an urban city like Boston be applicable in a rural city?
        
        Is it fair to judge the price of an individual home based on the characteristics 
        of the entire neighborhood?
        

Answer:
    
    Data collected in 1978 are actually pretty old. Many things happened ever since, 
    the inflation increased by 276.6%, city projects, perhaps extreme natural conditions
    (snowstorms, floods, etc.) that might have changed the preferences of the people.
    Nevertheless, one can argue that the prices in the housing market remained analogous
    to those at the time. It could be true that the prices only increased percentage-wise
    thus not being affected much by all these events, but only their value to have changed. 
    
    
    I think the data are reflecting the house prices, however there might be some
    valuable input ignored. For example, the age of the property, its condition,
    if there is an open space etc matter a lot in the price. Although we want fewer
    and as most descriptive factors as possible, a couple more factors would be great. 
    On the other hand, the current factors are still pretty descriptive if we admit that
    there is a positive correlation between lower class population and higher student/teacher
    ration and criminality. That makes a "criminality rate" factor redundant, for instance. 
    
    The model's output is quite reasonable, although not very robust. In particular, 
    compared to the max price value we computed above, it has approximately 8% of it. 
    Ideally I would like it to be around 3%-4% or less. 
    
    Depending on the size of the neighborhood, it is reasonable to estimate a range
    in which the price of an individual house should be in. We can then classify (cluster)
    house prices per basic neighborhood characteristics, but then it would be very bold
    to predict the price of an individual house, there it's likely this house to be an 
    exception (outlier).

    In general, if I was to sell such software to a real estate agent I would ask for
    more recent data -- data from the previous 5 year up to today. Then I will try to 
    work a bit more on the rebustness of it, and perhaps compute more metrics to make sure
    all is good. Finally I would test it for a month or two, and then deliver it.    
    
'''
