# Team Nine Final Project
# 
# 
# %%
############ IMPORTING LIBRARIES ############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import statsmodels.api as sm
# %%
############ IMPORTING DATA ############
data = pd.read_csv("Team9_data.csv")
data.head()
# %%
data.shape # 84,548 rows and 22 columns
# %%
data.describe()
# %%
data.nunique()

# %%
############ DATA CLEANING ############
data['TAX CLASS AT PRESENT'].value_counts()
# %%
class_sale_count = data['BUILDING CLASS AT TIME OF SALE'].value_counts()
class_sale_count
# %%
data["BOROUGH"].value_counts()
# %%
data["Unnamed: 0"].value_counts() # seems ambiguous
data = data.drop("Unnamed: 0", axis=1)
# %% 
# fixing whitespace in columns
newcols = ["borough", "neighborhood", "building_class_category", "tax_class_at_present", "block", "lot", "easement", "building_class_at_present", "address", "apartment_number", "zip_code", "residential_units", "commercial_units", "total_units", "land_square_feet", "gross_square_feet", "year_built", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price", "sale_date"]
data.columns = newcols
data.columns
# %% [markdown]
# There are missing values in the following variables:
# * TAX CLASS AT PRESET
# * EASE-MENT
# * BUILDING CLASS AT PRESENT
# * APARTMENT NUMBER
# * SALE PRICE
# 
# We should consider what do do with these missing values, like 
# replace them with the mode. Ease-ment should probably be dropped 
# because the column is mostly emtpy. Sale Price might indicate when 
# a property was not sold.
# %%
print("Number of na's in each column")
for col in data.columns:
    if data[col].dtype == str:
        data[col].str.strip()
        na = data[data[col] == ''].shape[0]
        na += data[data[col] == '-'].shape[0]
    else:
        na = data[data[col] == ' '].shape[0]
        na += data[data[col] == ' -  '].shape[0]
    # if data[col].dtype == str:
    #     na += data[data[col].str.contains("-")].shape[0]
    # I think there are other indicators for missing values that we need to find
    print(f"{col}: {na}")
    
data = data.drop("easement", axis=1)
# Decide how to fill the remaining columns

# %% Tax Class at Present
print(f"Value counts before fill:\n {data['tax_class_at_present'].value_counts()}")
# Can fill in the missing values with the mode
# data["tax_class_at_present"] = data["tax_class_at_present"].replace([" "], data["tax_class_at_time_of_sale"])

data["tax_class_at_present"] = np.where(data["tax_class_at_present"] == " ", data["tax_class_at_time_of_sale"], data["tax_class_at_present"])

# data['tax_class_at_present'].fillna(data["tax_class_at_time_of_sale"], inplace = True)
data['tax_class_at_present'].value_counts()
# %% Building Class at Present
print(f"Value coutns before fill: {data.building_class_at_present.value_counts}")


#%%
# year_built = 0 is an outlier, so we will drop these values.
print(len(data[data['year_built']==0]))
data = data[data['year_built']!=0]
data.year_built.isna()

# instead of year_built column, changing it to age by subtracting it by 2022 and dropping year_built column
data['age'] = 2022 - data['year_built']
data = data.drop("year_built", axis = 1)
column_move = data.pop("age")
data.insert(15, "age", column_move)

#%%
# converting sale_date to datetime datatype
data['sale_date'] = pd.to_datetime(data['sale_date'])

# converting land_square_feet and gross_square_feet to float datatype
data['land_square_feet'] = pd.to_numeric(data['land_square_feet'], errors='coerce')
data['gross_square_feet'] = pd.to_numeric(data['gross_square_feet'], errors='coerce')

#%%
data = data.drop(['land_square_feet'], axis = 1)  # dropping land_square_feet 
data = data.dropna(subset=['gross_square_feet'])  # dropping NA values rows from gross_square_feet column



#%%
print(data.select_dtypes(['object']).columns)
# Index(['neighborhood', 'building_class_category', 'tax_class_at_present',
#        'building_class_at_present', 'address', 'apartment_number',
#        'land_square_feet', 'gross_square_feet',
#        'building_class_at_time_of_sale', 'sale_price', 'sale_date'],
#       dtype='object')

print(data.select_dtypes(['int64']).columns)
# Index(['borough', 'block', 'lot', 'zip_code', 'residential_units',
#        'commercial_units', 'total_units', 'age', 'tax_class_at_time_of_sale'],
#       dtype='object')

# converting few object type variables to category variable
obj_to_category = ['building_class_category', 'tax_class_at_present', 'building_class_at_present', 'building_class_at_time_of_sale']

for colname in obj_to_category:
    data[colname] = data[colname].astype('category') 

# converting few int type variables to category variable
int_to_category = ['borough', 'tax_class_at_time_of_sale']

for colname in int_to_category:
    data[colname] = data[colname].astype('category')


# %%
############ EXPLORATORY DATA ANALYSIS ############
# Starting with sold homes - dropping those with " -  ", 0 or 10
data_sold = data[data["sale_price"].str.contains(" -  ") == False]
data_sold = data_sold[data_sold["sale_price"] != "0"]
data_sold = data_sold[data_sold["sale_price"] != "10"]
data_sold.shape

# %%
data_sold["sale_price"] = data_sold["sale_price"].astype(float)

# %%
# Creating Price per Square Foot
# try:
#     data_sold["price_per_sqft"] = data_sold["sale_price"] / data["gross_square_feet"].astype(float)
data["sale_price"] = data_sold["sale_price"].astype(float)

# except:
#     data_sold["price_per_sqft"] = None
# %%
data_borough = data_sold[["borough", "sale_price"]].groupby(["borough"]).mean()
labels = data_borough.index.values
values = data_borough['sale_price'].values
data_borough

#%%
# data = data.dropna(subset=['sale_price'])  # dropping NA values rows from sale_price column
data = data.drop(['building_class_at_present', 'tax_class_at_present'], axis = 1) 
# data.isna().sum()


# %%
############ DATA VISUALIZATION ############
plt.bar(labels, values)
# ax.set_xticklabels(("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"))
plt.ylabel("Average Sale Price (Millions)")
plt.xlabel("Borough")
plt.title("Average Sale Price by Borough")
plt.show()

# %%
plt.hist(data_sold.sale_price, bins = 30, edgecolor = 'black')
plt.xlabel("Price of House")
plt.xlim(right=1000000)
plt.xlim(left=0)
plt.ylabel("Frequency")
plt.title("Frequency of House Sale Prices")
plt.show()

# %%
data_sqft = data_sold[data_sold["gross_square_feet"] != " -  "]
sns.lmplot(x = "gross_square_feet", y = "sale_price", data = data_sqft)
plt.show()

# %% [markdown]
# Variables we will use in our model:
# * borough
# * building_class_category
# * zip_codes
# * total_units
# * age
# * tax_class_at_time_of_sale
# * building_class_at_time_of_sale
# * sale_price.
# %%
############ MODEL BUILDING ############
# Linear Regression
from statsmodels.formula.api import ols

form = "SALE PRICE ~ BOROUGH + NEIGHBORHOOD + BUILDING CLASS CATEGORY + TAX CLASS AT PRESENT + BLOCK + LOT + BUILDING CLASS AT PRESENT + ZIP CODE + RESIDENTIAL UNITS + COMMERCIAL UNITS"
# modelPrice = ols(formula=form, data=data) # Syntax error

# modelPrice = ols(formula = "SALE_PRICE ~ BOROUGH", data=data) - not working
# modelPrice.summary()
# %%
# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
# %%
data_sold_features = data_sold[["borough", "building_class_category", "zip_code", "total_units", "age", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price"]]

cats = ["building_class_category", "tax_class_at_time_of_sale", "building_class_at_time_of_sale"]

for cat in cats:
    labelendoder = LabelEncoder()
    data_sold_features[cat] = labelendoder.fit_transform(data_sold_features[cat])

data_sold_features.head()
# %%
## ONE HOT ENCODER NOT WORKING - Try to troubleshoot later, will use label encoder instead
dataPreprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(), ["building_class_category"
                                          , "tax_class_at_time_of_sale", "building_class_at_time_of_sale"
                                          ])
    ], verbose_feature_names_out=False, remainder="passthrough",
).fit(data_sold_features)

dataPreprocessor.get_feature_names_out()

data_transformed = pd.DataFrame(dataPreprocessor.fit_transform(data_sold_features),    
                                columns=dataPreprocessor.fit(data_sold_features).get_feature_names_out()
                                )

data_transformed.head()
# %%
X = data_sold_features.drop(["sale_price"], axis=1)
y = data_sold_features['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=55)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %% Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
lr.score(X_test, y_test)
# %% SVR

# %% Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt.score(X_test, y_test)
# %% Random Forest
rf = RandomForestRegressor(random_state=10)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
rf.score(X_test, y_test)
# %%
