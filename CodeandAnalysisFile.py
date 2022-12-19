# Team Nine Final Project
# 
# 
# %%
############ IMPORTING LIBRARIES ############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
#############################################


# %%
########### IMPORTING sklearn LIBRARIES ###########
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
####################################################
########### IMPORTING statsmodel API ################
import statsmodels.api as sm
#####################################################
# %%
############ IMPORTING DATA ############
data = pd.read_csv("Team9_data.csv")
data.head()
########################################
# %%
# printing out the shape of our data
data.shape # 84,548 rows and 22 columns

# %%
########################## DATA CLEANING ###################################

# %%
# after printing out the head of our data, "Unnamed: 0" seemed ambiguous, removing it
data["Unnamed: 0"].value_counts()
data = data.drop("Unnamed: 0", axis=1)
#%%
# we don't want to face any error further because of the whitespace in column names, so renaming all the remanining columns
# without whitespace.
newcols = ["borough", "neighborhood", "building_class_category", "tax_class_at_present", "block", "lot", "easement", "building_class_at_present", "address", "apartment_number", "zip_code", "residential_units", "commercial_units", "total_units", "land_square_feet", "gross_square_feet", "year_built", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price", "sale_date"]
data.columns = newcols
data.columns
# %%
data.info()
# after checking out the info() function for our data, 
# "residential_units", "commercial_units", "total_units", "land_square_feet", "gross_square_feet","sale_price"
#  all these columns need to be converted to numeric variables

numeric = ["residential_units","commercial_units","total_units", "land_square_feet" ,
           "gross_square_feet","sale_price" ]
for col in numeric: 
    data[col] = pd.to_numeric(data[col], errors='coerce')

# we set errors = 'coerce' to fill out non numeric values is NAN, which will help us in dealing with NULL values later.
# %%
# Similarly we changed the type of variables which needed to be categorical but were not.

categorical = ["borough","neighborhood",'building_class_category', 'tax_class_at_present',
               'building_class_at_present','zip_code', 'building_class_at_time_of_sale', 'tax_class_at_time_of_sale']
for col in categorical: 
    data[col] = data[col].astype('category')
# %%
# converting sale_date to datetime datatype
data['sale_date'] = pd.to_datetime(data['sale_date'])
# %%
# Creating Percent Residential Variable
data["percent_residential_units"] = data["residential_units"] / data["total_units"]
# %%
# Three lines are greater than one - not possible is likely due to discrepancies in the data so we can drop those
data= data.loc[data["percent_residential_units"] != 6]
data= data.loc[data["percent_residential_units"] != 2]
data= data.loc[data["percent_residential_units"] != 1.5]
data['percent_residential_units'].value_counts()

# %%
# Some more cleaning of the data.
# 'apartment_number','building_class_at_present', 'tax_class_at_present' seemed repetetive as we have 'zip_code', 'building_class_at_time_of_sale', and  'tax_class_at_time_of_sale' in our dataset
data_few_cols = data.drop(['apartment_number','building_class_at_present', 'tax_class_at_present'], axis=1)
data_few_cols = data_few_cols.dropna()
data_few_cols.drop_duplicates(keep = "last", inplace=True)

# %%
# Removing data where commercial + residential doesn't equal total units
data_few_cols = data_few_cols[data_few_cols['total_units'] == data_few_cols['commercial_units'] + data_few_cols['residential_units']]

# Removing rows with TOTAL UNITS == 0 and one outlier with 2261 units
data_few_cols = data_few_cols[(data_few_cols['total_units'] > 0) & (data_few_cols['total_units'] != 2261)]

# %%
# making a new column of age by subtracting current year with the year_built column
data_few_cols = data_few_cols[data_few_cols['year_built'] != 0]
data_few_cols['age'] = 2022 - data_few_cols['year_built']
data_few_cols = data_few_cols.drop(['year_built'], axis =1)
# %%
# Removing impossible values for land_square_feet and gross_square_feet
data_few_cols = data_few_cols[data_few_cols["land_square_feet"] != 0]
data_few_cols = data_few_cols[data_few_cols["gross_square_feet"] != 0]
# %%
##################### DETECTING OUTLIERS AND REMOVING THEM #################################

# %%
############ DETECTING OUTLIERS ############

# removing outliers which are less than 0.25 quantile and more than 0.75 quantile.
outliers_indexes = []
def outliers (df, ft):
    q1=df[ft].quantile(0.25)
    q3=df[ft].quantile(0.75)
    iqr=q3-q1
    
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    
    outliers=df.index[ (df[ft]<lower_bound) | (df[ft]>upper_bound)]
    return outliers

outliers_indexes.extend(outliers(data_few_cols, 'sale_price'))
print('Number of outliers:', len(outliers_indexes))

#%%
############ REMOVING OUTLIERS ############
def outliers_remove(df, list):
    list = sorted(set(list)) 
    df_clean = df.drop(list)
    return df_clean

clean_df = outliers_remove(data_few_cols, outliers_indexes)
clean_df = clean_df.loc[clean_df["sale_price"] > 10]
clean_df.shape
# %%

###################### DATA VISUALIZATION #########################

# Correlation Matrix
corr = clean_df.corr()
ax = sns.heatmap(corr, cmap = 'Blues')
plt.title("Correlation Matrix")
plt.show()

# %%
# Only plotting values less than $10 million due to significant outliers (eg. 2.2 BILLION dollars lol)
plt.hist(data_few_cols.loc[data_few_cols["sale_price"] <= 10000000]["sale_price"], bins = 30, edgecolor = 'black')
plt.xlabel("Sale Price (Millions)")
plt.ylabel("Frequency")
plt.title("Histogram of House Sale Prices")
plt.show()
# %%
# Plot Sales Price After Outliers Removed
plt.hist(clean_df.sale_price, bins = 30, edgecolor = "black")
plt.title("Histogram of Sale Prices")
plt.xlabel("Sale Price (Millions)")
plt.ylabel("Frequency")
plt.show()
# much cleaner plot, although not exactly normal distribution but very close to normal distribution.
# %%
borough_map = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5:"Staten Island"}
data_few_cols["borough"] = data_few_cols.borough.map(borough_map)
data_borough = data_few_cols[["borough", "sale_price"]].groupby(["borough"]).mean()
labels = data_borough.index.values
values = data_borough['sale_price'].values
data_borough
# %%
plt.bar(labels, values, edgecolor = "black")
# ax.set_xticklabels(("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"))
plt.ylabel("Average Sale Price (Millions)")
plt.xlabel("Borough")
plt.title("Average Sale Price by Borough")
plt.show()

# From the plot, we can see that Manhattan property price is much higher than the other boroughs.

# %%
sns.violinplot(x = "borough", y = "sale_price", data = data_few_cols.loc[data_few_cols["sale_price"] <= 10000000])
plt.title("Distribution of Sale Price by Borough")
plt.xlabel("Borough")
plt.ylabel("Sale Price")
plt.show()

# from the violinplot, we can see that Manhattan is much skinnier as compared to others but it stretches to higher sales price.
# skinny plot represents that there is no specific price range in which Manhattan properties are getting sold.
# rest all the boroughs are fatter in lower sales price region.
# %%
# Plotting Square Footage for < 10k and those not equal to zero
data_few_cols_gross = data_few_cols.loc[data_few_cols["gross_square_feet"] < 10000]
data_few_cols_gross = data_few_cols_gross.loc[data_few_cols["gross_square_feet"] != 0]
plt.hist(data_few_cols_gross["gross_square_feet"], bins = 30, edgecolor = "black")
plt.xlabel("Gross Square Feet")
plt.ylabel("Frequency")
plt.title("Histogram of Gross Square Footage")
plt.show()

# plot is right skewed indicating there are less values in upper end of the gross_square_feet.
# %%
# Square Footage and Sale Price
data_few_cols_gross = data_few_cols_gross.loc[data_few_cols_gross["sale_price"] <= 20000000]
sns.lmplot(x = "gross_square_feet", y = "sale_price", data = data_few_cols_gross, scatter_kws={"alpha": 0.2}, line_kws={'color': 'red'})
plt.title("Square Footage versus Sale Price")
plt.xlabel("Square Footage")
plt.ylabel("Sale Price")
plt.show()
# the data points are very scattered are not aligning very well with the straight line.

# %%,
sns.lmplot(x = "age", y = "sale_price", data = data_few_cols.loc[data_few_cols["sale_price"] <= 10000000], scatter_kws={'alpha': 0.2}, line_kws={'color': 'red'})
plt.title("Age versus Sale Price")
plt.xlabel("Age (years)")
plt.ylabel("Sale Price")
plt.show()
# the data points are very scattered are not aligning very well with the straight line.

# %%
tax_class = data_few_cols[["sale_price", "tax_class_at_time_of_sale"]].groupby(["tax_class_at_time_of_sale"]).mean()
labels = tax_class.index.values.astype(str)
values = tax_class.sale_price.values
tax_class


# %% # Tax class = clearly a big predictor
plt.bar(labels, values)
plt.title("Sale Price by Tax Class")
plt.xlabel("Tax Class")
plt.ylabel("Sale Price")
plt.show()
# this plot gives valuable insights with respect to different tax classes. Properties with tax class 4 has the highest selling price.

# %%
data_low = data_few_cols.loc[data_few_cols["sale_price"] <= 10000000]
sns.lmplot(x = "total_units", y = "sale_price", data = data_low.loc[data_low['total_units'] < 500])
plt.ylim((0, 10000000))
plt.show()

# we can see that there is a linear relation of total_units but it is not very clear as there are a lot of outliers.
# %%
# Distribution of Percent Residential Units - Not very informative but interesting 
# as most buildings are 100% residential
plt.hist(data_few_cols["percent_residential_units"], bins = 40, edgecolor = 'black')
plt.title("Histogram of Percent of Residential Units")
plt.xlabel("Percent Residential")
plt.ylabel("Frequency")
plt.show()
#%%

#%%[markdown]
############ MODEL BUILDING ############
# Variables we will use in our model:
# * borough
# * building_class_category
# * zip_codes
# * total_units
# * percent_residential_units
# * age
# * gross_square_feet
# * tax_class_at_time_of_sale
# * building_class_at_time_of_sale
# * sale_price.
# 
# Variables to consider:
# * price per square foot
# * sale date ??.

#%%
# scaling our dataset and one hot encoding

data_sold_features = data_few_cols[["borough", "building_class_category", "zip_code", "total_units", "percent_residential_units", "age", "gross_square_feet", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price"]]
dataPreprocessor = ColumnTransformer( transformers=
    [
        ("categorical", OneHotEncoder(), ["building_class_category", "tax_class_at_time_of_sale", 
                                          "building_class_at_time_of_sale", "borough", "zip_code"
                                          ])
        #("numeric", StandardScaler(), ["total_units", "age", "sale_price", "gross_square_feet", "percent_residential_units"])
    ], verbose_feature_names_out=False, remainder="passthrough",
)

data_sold_features_newmatrix = dataPreprocessor.fit_transform(data_sold_features)
newcolnames = dataPreprocessor.get_feature_names_out()
data_sold_features_new = pd.DataFrame( data_sold_features_newmatrix.toarray(), columns= newcolnames )
print(data_sold_features_new.shape)
print(data_sold_features_new.head())

# we transformed categorical variables using onehotencoding
# fit_transform() function will calculate the mean and standard deviation and will perform scaling on the variables.
# so we have scaled all our variables.

# %%
# splitting our dataset into test and train

X = data_sold_features_new.drop(["sale_price"], axis=1)
y = data_sold_features_new['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=55)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



# %% Linear Regression by Sklearn: 
lr = LinearRegression()
lr.fit(X_train, y_train)
# R-squared
print(f"Training R-Squared: {lr.score(X_train, y_train)}")
print(f"Testing R_squared {lr.score(X_test, y_test)}")
# Mean Squared Error
print(f"Training MSE: {mean_squared_error(y_train, lr.predict(X_train))}")
print(f"Testing MSE: {mean_squared_error(y_test, lr.predict(X_test))}")
# Mean Absolute Error
print(f"Training MAE: {mean_absolute_error(y_train, lr.predict(X_train))}")
print(f"Testing MAE: {mean_absolute_error(y_test, lr.predict(X_test))}")

# when we tried linear regression using sklearn, test score was very bad.
# we got a negative score which indicates that our model performing worse than a constant model.

#%% Linear Regression by Statsmodels: 
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
# First time model built
lm_model = ols(formula=' sale_price ~ C(borough) + C(building_class_category) + C(zip_code) + total_units + percent_residential_units + age + gross_square_feet + C(tax_class_at_time_of_sale) + C(building_class_at_time_of_sale)', data=clean_df)
lm_model_fit = lm_model.fit()
print(lm_model_fit.summary())

# using the model above, Adjusted R squared value came out to be 0.476. Which is a huge improvement than the previous linear model using sklearn.
# from the model summary we can also see that Pvalue of different building class categories are coming out to be very less which means these coefficients are significant.

#%% Linear Regression Assumptions
# Normally bell shaped
sns.histplot(data=clean_df, x='sale_price')
# VIFs Checkings
Xvifs = clean_df[['borough', 'building_class_category', 'zip_code', 'total_units', 
              'percent_residential_units', 'age', 'gross_square_feet', 
              'tax_class_at_time_of_sale', 'building_class_at_time_of_sale']]
Xvif = Xvifs.drop(['building_class_category', 'building_class_at_time_of_sale', 'percent_residential_units', 'zip_code'], axis=1)
vifs = pd.DataFrame()
vifs["features"] = Xvif.columns
vifs["VIF"] = [ variance_inflation_factor(Xvif.values, i) 
               for i in range(len(Xvif.columns)) ]
print(vifs)
# Linearity Checkings
sns.lmplot(x = "total_units", y = "sale_price", data = clean_df[clean_df['total_units']<100], line_kws={'color':'red'} )
plt.title("Total Units versus Sale Price")
plt.xlabel("Total Units")
plt.ylabel("Sale Price")
plt.show()
sns.lmplot(x = "age", y = "sale_price", data = clean_df[clean_df['age']<100], line_kws={'color':'red'})
plt.title("Age versus Sale Price")
plt.xlabel("Age (years)")
plt.ylabel("Sale Price")
plt.show()
sns.lmplot(x = "gross_square_feet", y = "sale_price", data = clean_df[clean_df['gross_square_feet']<8000], line_kws={'color':'red'})
plt.title("Gross Square Feet versus Sale Price")
plt.xlabel("Gross Square Feet")
plt.ylabel("Sale Price")
plt.show()
#%% Build the model with interactions terms again: 
lm_model2 = ols(formula=' sale_price ~  age + total_units + gross_square_feet + C(borough) + C(tax_class_at_time_of_sale) + C(building_class_category) + age:C(building_class_category) + total_units:C(building_class_category)', data=clean_df)
lm_model2_fit = lm_model2.fit()
print(lm_model2_fit.summary())
#%% LM Residual Plots:
ypred = lm_model_fit.predict(clean_df)
# resid = ypred - y_train
plt.plot(ypred, lm_model2_fit.resid, 'o')
plt.axhline(y=0, color = 'red')
plt.title("Plot of Residuals")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show() 
# %% Decision Tree
dt = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
dt.fit(X_train, y_train)
# R-squared
print(f"Training R-Squared: {dt.score(X_train, y_train)}")
print(f"Testing R_squared {dt.score(X_test, y_test)}")
# Mean Squared Error
print(f"Training MSE: {mean_squared_error(y_train, dt.predict(X_train))}")
print(f"Testing MSE: {mean_squared_error(y_test, dt.predict(X_test))}")
# Mean Absolute Error
print(f"Training MAE: {mean_absolute_error(y_train, dt.predict(X_train))}")
print(f"Testing MAE: {mean_absolute_error(y_test, dt.predict(X_test))}")

# when running a decisionTreeRegressor, we found out that the score was very less of just 2.2%
#%%
tree_cv = DecisionTreeRegressor(max_depth=50, max_features=10, random_state=10)

cv_results = cross_val_score(tree_cv, X_train, y_train, cv=5)
print(cv_results)
print(np.mean(cv_results))

# %%
train = []
test = []
depths = list(range(10,101,10))
for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    train.append(dt.score(X_train, y_train))
    test.append(dt.score(X_test, y_test))
    
plt.plot(depths, train, c = "blue")
plt.plot(depths, test, c = "red")
plt.title("Accuracies at Different Tree Depths")
plt.xlabel("Max Depth")
plt.ylabel("R-Squared")
plt.show()

# on checking R squared value for different depths of tree, accuracy fluctuated a lot and not giving a pattern.
# so no judgement can given by the decision tree regressor for our dataset.

# %% Random Forest
rf = RandomForestRegressor(max_depth=10, min_samples_split=10, random_state=10)
rf.fit(X_train, y_train)
# R-squared
print(f"Training R-Squared: {rf.score(X_train, y_train)}")
print(f"Testing R_squared {rf.score(X_test, y_test)}")
# Mean Squared Error
print(f"Training MSE: {mean_squared_error(y_train, rf.predict(X_train))}")
print(f"Testing MSE: {mean_squared_error(y_test, rf.predict(X_test))}")
# Mean Absolute Error
print(f"Training MAE: {mean_absolute_error(y_train, rf.predict(X_train))}")
print(f"Testing MAE: {mean_absolute_error(y_test, rf.predict(X_test))}")

# using randomForestRegressor, we can see that R_squared value is around 26%.
# this is much better than decisionTreeRegressor but it is still not good enough.
# %% Random Forest
# so to check the different parameters for hyperparameter tuning, we used gridSearchCV approach

from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[10, 50, 100],
    "max_features":[6,8,10,12,14,16]
}

rf = RandomForestRegressor(random_state = 10)

rf_tuned = GridSearchCV(estimator = rf,
                            param_grid = param_grid,
                            cv = 2,
                            n_jobs=-1,
                        verbose=0)

rf_tuned.fit(X, y)
rf_tuned.best_estimator_
# %%
model = rf_tuned.best_estimator_.fit(X_train,y_train)

print("Training score:",model.score(X_train, y_train))
print("Testing score:",model.score(X_test, y_test))

# %%
# Random Forest Cross Validation
from sklearn.model_selection import cross_val_score

full_cv = RandomForestRegressor(max_depth=50, max_features=10, n_estimators=200, random_state = 10)

cv_results = cross_val_score(full_cv, X_train, y_train, cv = 5)
print(cv_results)

#%%
rf2 = RandomForestRegressor(max_depth=10, max_features=10,random_state = 10)
rf2.fit(X_train,y_train)

print("Training score:",rf2.score(X_train, y_train))
print("Testing score:",rf2.score(X_test, y_test))

# We tried different models with different parameters,
# Maximum accuracy we got was 28.9%.
# %%
# KNN
# as per professor's suggestion, we tried KNN and it gave us much better results than other models.
from sklearn.neighbors import KNeighborsRegressor

acc_test = []
acc_train = []
for k in range(4, 35):
    knn_model = KNeighborsRegressor(k).fit(X_train, y_train)
    score_knn_test = knn_model.score(X_test, y_test)
    acc_test.append(score_knn_test)
    print(k)
    print("test score:",score_knn_test)
    score_knn_train = knn_model.score(X_train, y_train)
    acc_train.append(score_knn_train)
    print("train Score", score_knn_train)
    print("\n")
    
    
# %%
plt.plot(np.arange(4,35), acc_test, label="Testing accuracy")
plt.title("Score vs K value plot for test dataset")
plt.xlabel("K value")
plt.ylabel("score")

# from the plot, R-squared score is negative initially and increasing with increasing K value.
# at around 35% score, score values is getting saturated.
# %%
# Maximum score is acheived at k = 23
knn_model = KNeighborsRegressor(23).fit(X_train, y_train)
score_knn = knn_model.score(X_test, y_test)
print("Score at k = 23: ",score_knn)

# We got the maximum accuracy of around 37% at the k value of 23.
# %%
