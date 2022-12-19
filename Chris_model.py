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
#plt.hist()

# %%
# Correlation Matrix
clean_df.corr()
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
# * percent_residential_units
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

# %%
########### Building model with Sklearn ############
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

# %% Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

#%%
# R-squared
print(f"Training R-Squared: {lr.score(X_train, y_train)}")
print(f"Testing R_squared {lr.score(X_test, y_test)}")

#%%
# Mean Squared Error
print(f"Training MSE: {mean_squared_error(y_train, lr.predict(X_train))}")
print(f"Testing MSE: {mean_squared_error(y_test, lr.predict(X_test))}")

#%%
# Mean Absolute Error
print(f"Training MAE: {mean_absolute_error(y_train, lr.predict(X_train))}")
print(f"Testing MAE: {mean_absolute_error(y_test, lr.predict(X_test))}")




# %%
########### Building model with statsmodel ############
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%
### Adding one more column 'season'
clean_df['sale_date'] = pd.to_datetime(clean_df['sale_date'])
clean_df['sale_month'] = clean_df['sale_date'].dt.month
# clean_df['month'].describe()

def season(month): 
    if month==12 or month==1 or month==2:
        season = 'Winter'
    elif month==9 or month==10 or month==11:
        season = 'Autumn'
    elif month==6 or month==7 or month==8:
        season = 'Summer'
    else:
        season = 'Spring'
    return season

clean_df['Season'] = clean_df['sale_month'].apply(season)
clean_df['Season'] = clean_df['Season'].astype('category')
clean_df['Season'].info()
print('')
clean_df['Season'].value_counts()

# %%
### First build
lm_model = ols(formula=' sale_price ~ C(Season) + C(borough) + C(building_class_category) + C(zip_code) + total_units + percent_residential_units + age + gross_square_feet + C(tax_class_at_time_of_sale) + C(building_class_at_time_of_sale)', data=clean_df)
lm_model_fit = lm_model.fit()
print(lm_model_fit.summary())

# %%
########## Model Assumptions  ############
## Normally bell shaped ##
sns.histplot(data=clean_df, x='sale_price')

#%%
## VIFs Checkings ##
Xvifs = clean_df[['borough', 'building_class_category', 'zip_code', 'total_units', 
              'percent_residential_units', 'age', 'gross_square_feet', 
              'tax_class_at_time_of_sale', 'building_class_at_time_of_sale', 'Season']]

Xvif = Xvifs.drop(['building_class_category', 'building_class_at_time_of_sale', 'Season', 'percent_residential_units', 'zip_code'], axis=1)
vifs = pd.DataFrame()
vifs["features"] = Xvif.columns
vifs["VIF"] = [ variance_inflation_factor(Xvif.values, i) 
               for i in range(len(Xvif.columns)) ]
print(vifs)

#try:
#    vifs = pd.DataFrame()
#    vifs["features"] = Xvifs.columns
#    vifs["VIF"] = [ variance_inflation_factor(Xvifs.values, i) 
#                   for i in range(len(Xvifs.columns)) ]
#    print(vifs)
#except TypeError:
#    Xvif = clean_df[['borough', 'zip_code', 'total_units', 
#              'percent_residential_units', 'age', 'gross_square_feet', 
#              'tax_class_at_time_of_sale']]
#    vifs2 = pd.DataFrame()
#    vifs2["features"] = Xvif.columns
#    vifs2["VIF"] = [ variance_inflation_factor(Xvif.values, i) 
#                   for i in range(len(Xvif.columns)) ]
#    print(vifs)

#%%
## Linearity Checkings ##
sns.lmplot(x = "total_units", y = "sale_price", data = clean_df[clean_df['total_units']<100], line_kws={'color':'red'} )
plt.ylim((0, 10000000))
plt.show()
sns.lmplot(x = "age", y = "sale_price", data = clean_df[clean_df['age']<100], line_kws={'color':'red'})
plt.show()
sns.lmplot(x = "gross_square_feet", y = "sale_price", data = clean_df[clean_df['gross_square_feet']<8000], line_kws={'color':'red'})
plt.show()

# %%
X = clean_df.drop(["sale_price"], axis=1)
y = clean_df['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=55)

X_train['sale_price'] = y_train

#%%
lm_model = ols(formula=' sale_price ~  C(Season) + C(building_class_category) + age + total_units + C(tax_class_at_time_of_sale) + C(borough) + gross_square_feet + age:C(building_class_category) + total_units:C(building_class_category)', data=X_train)
lm_model_fit = lm_model.fit()
print(lm_model_fit.summary())
# + C(building_class_at_time_of_sale) 
#%%
pre_v_act = pd.DataFrame( columns=['Predict'], data=lm_model_fit.predict(X_test)) 
pre_v_act['Actual'] = y_test
print(pre_v_act.shape)
print(pre_v_act.head())

## Average Biased
pre_v_act['Differences'] = pre_v_act.apply(lambda row: abs( (row[0]-row[1])/row[1]), axis=1)

print('With outliers:', pre_v_act['Differences'].describe())
clean = pre_v_act[pre_v_act['Differences']<1]
print('Without outliers:', clean['Differences'].describe())
print('')
print(clean['Differences'].mean())


#%% [markdown]
# From the summary, try to get as much info as we can
# Df Residuals (# total observations minus Df Model minus 1)
# Df Model (# of x variables)
# R-squared, what does that mean?
# Adj R-squared
# F-statistics
# Prob (F-statistics), ie. p-value for F-statistics
# Log-Likelihood
# AIC (model eval)
# BIC (model eval)

# coef
# std err
# t
# P>|t|, aka p-value for the coefficient significance
# 95% confidence intervals

# Omnibus - close to zero means residuals are normally distributed
# Prob(Omnibus) - close to 1 means residuals are normally distributed
# skew (positive is right tailed, negative is left)
# Kurtosis (tailedness, normal dist = 3, less than 3 is fatter tail, and flat top.)

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
# Variables we didn't use:
# * easement: Too
# * neighborhood, block, lot, address：These are all 
# * land_square_feeet：Overlap information with gross_square_feet and we prefer more on gross_square_feet

# Variables to consider:
# * percent_residential_units
# * price per square foot
# * sale date ??.



#%%
##### Overlapping content celeted from main file #####
# Linear Regression
from statsmodels.formula.api import ols

form = "sale_price ~ C(borough) + C(building_class_category) + C(zip_code) + total_units + percent_residential_units + age + gross_square_feet + C(tax_class_at_time_of_sale) + C(building_class_at_time_of_sale)"

modelPrice = ols(formula=form, data=clean_df).fit()
modelPrice.summary()