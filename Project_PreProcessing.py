# Team Nine Final Project
# 
# 
# %%
############ IMPORTING LIBRARIES ############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


# %%
# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
# %%
############ IMPORTING DATA ############
data = pd.read_csv("Team9_data.csv")
data.head()
# %%
data.shape # 84,548 rows and 22 columns

#%%
# %%
data["Unnamed: 0"].value_counts() # seems ambiguous
data = data.drop("Unnamed: 0", axis=1)
#%%
# fixing whitespace in columns
newcols = ["borough", "neighborhood", "building_class_category", "tax_class_at_present", "block", "lot", "easement", "building_class_at_present", "address", "apartment_number", "zip_code", "residential_units", "commercial_units", "total_units", "land_square_feet", "gross_square_feet", "year_built", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price", "sale_date"]
data.columns = newcols
data.columns

#%%
# %%
numeric = ["residential_units","commercial_units","total_units", "land_square_feet" ,
           "gross_square_feet","sale_price" ]
for col in numeric: 
    data[col] = pd.to_numeric(data[col], errors='coerce')
# %%
categorical = ["borough","neighborhood",'building_class_category', 'tax_class_at_present',
               'building_class_at_present','zip_code', 'building_class_at_time_of_sale', 'tax_class_at_time_of_sale']
for col in categorical: 
    data[col] = data[col].astype('category')
    
# %%
# Creating Percent Reseidential Variable
data["percent_residential_units"] = data["residential_units"] / data["total_units"]
# Three lines are greater than one - not possible is likely due to discrepencies in the data so we can drop those
data= data.loc[data["percent_residential_units"] != 6]
data= data.loc[data["percent_residential_units"] != 2]
data= data.loc[data["percent_residential_units"] != 1.5]
data['percent_residential_units'].value_counts()
#%%
data_few_cols = data.drop(['apartment_number','building_class_at_present', 'tax_class_at_present'], axis=1)
#%%
data_few_cols = data_few_cols.dropna()
# %%
# %%
data_few_cols.drop_duplicates(keep = "last", inplace=True)
# %%
# Removing data where commercial + residential doesn't equal total units
data_few_cols = data_few_cols[data_few_cols['total_units'] == data_few_cols['commercial_units'] + data_few_cols['residential_units']]

# %%
# Removing rows with TOTAL UNITS == 0 and one outlier with 2261 units
data_few_cols = data_few_cols[(data_few_cols['total_units'] > 0) & (data_few_cols['total_units'] != 2261)]

# %%
data_few_cols = data_few_cols[data_few_cols['year_built'] != 0]
data_few_cols['age'] = 2022 - data_few_cols['year_built']
# %%
data_few_cols = data_few_cols.drop(['year_built'], axis =1)
# %%
data_few_cols = data_few_cols[data_few_cols["land_square_feet"] != 0]
data_few_cols = data_few_cols[data_few_cols["gross_square_feet"] != 0]
# %%
############ DETECTING OUTLIERS ############
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
# Plot Sales Price After Outliers Removed
plt.hist(clean_df.sale_price, bins = 30, edgecolor = "black")
plt.title("Histogram of Sale Prices")
plt.xlabel("Sale Price (Millions)")
plt.ylabel("Frequency")
plt.show()
# %%
# Correlation Matrix
corr = clean_df.corr()
ax = sns.heatmap(corr, cmap = 'Blues')
plt.title("Correlation Matrix")
plt.show()
# %% [markdown]
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
# %%
############ MODEL BUILDING ############
# Linear Regression
from statsmodels.formula.api import ols

form = "sale_price ~ C(borough) + C(building_class_category) + C(zip_code) + total_units + percent_residential_units + age + gross_square_feet + C(tax_class_at_time_of_sale) + C(building_class_at_time_of_sale)"

modelPrice = ols(formula=form, data=clean_df).fit()
modelPrice.summary()

# %%
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

# %%
X = data_sold_features_new.drop(["sale_price"], axis=1)
y = data_sold_features_new['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=55)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# %% Linear Regression
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
# %%

# %% Random Forest
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

# %%
# Decision Tree
tree_cv = DecisionTreeRegressor(max_depth=50, max_features=10, random_state=10)

cv_results = cross_val_score(tree_cv, X_train, y_train, cv=5)
print(cv_results)
print(np.mean(cv_results))
# %%
