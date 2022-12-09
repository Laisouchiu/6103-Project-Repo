# Team Nine Final Project
# 
# 
# %%
## IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import statsmodels.api as sm

# %%
## IMPORTING DATA
data = pd.read_csv("Team9_data.csv")
data.head()
# %%
data.shape # 84,548 rows and 22 columns
# %%
data.describe()
# %%
data.nunique()

# %%
data["BOROUGH"].value_counts()
# %%
data["Unnamed: 0"].value_counts() # seems ambiguous
data = data.drop("Unnamed: 0", axis=1)
# %% fixing whitespace in columns
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
    na = data[data[col] == ' '].shape[0]
    na += data[data[col] == ' -  '].shape[0]
    if data[col].dtype == str:
        na += data[data[col].str.contains("-")].shape[0]
    # I think there are other indicators for missing values that we need to find
    print(f"{col}: {na}")
    
data = data.drop("easement", axis=1)
# Decide how to fill the remaining columns

# %% Tax Class at Present
print(f"Value counts before fill:\n {data['tax_class_at_present'].value_counts()}")
# Can fill in the missing values with the mode
data.replace(" ", np.nan)

data['tax_class_at_present'].fillna(data["tax_class_at_time_of_sale"], inplace = True)
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

# %% EDA
# Starting with sold homes - dropping those with " -  ", 0 or 10
data_sold = data[data["sale_price"].str.contains(" -  ") == False]
data_sold = data_sold[data_sold["sale_price"] != "0"]
data_sold = data_sold[data_sold["sale_price"] != "10"]
data_sold.shape

# %%
data_sold["sale_price"] = data_sold["sale_price"].astype(float)
data["sale_price"] = data_sold["sale_price"].astype(float)

# %%
data_borough = data_sold[["borough", "sale_price"]].groupby(["borough"]).mean()
labels = data_borough.index.values
values = data_borough['sale_price'].values
data_borough
# %%
# DATA VISUALIZATION
plt.bar(labels, values)
# ax.set_xticklabels(("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"))
plt.ylabel("Average Sale Price (Millions)")
plt.xlabel("Borough")
plt.title("Average Sale Price by Borough")
plt.show()

# %%
plt.hist(data.sale_price, bins = 20, edgecolor = 'black')
plt.xlabel("Price of House")
plt.ylabel("Frequency")
plt.title("Frequency of House Sale Prices")
plt.show()

# %%
data_sqft = data_sold[data_sold["gross_square_feet"] != " -  "]
sns.lmplot(x = "gross_square_feet", y = "sale_price", data = data_sqft)
plt.show()

#%%
# converting sale_date to datetime datatype
data['sale_date'] = pd.to_datetime(data['sale_date'])

# converting land_square_feet and gross_square_feet to float datatype
data['land_square_feet'] = pd.to_numeric(data['land_square_feet'], errors='coerce')
data['gross_square_feet'] = pd.to_numeric(data['gross_square_feet'], errors='coerce')

#%%
data = data.drop(['land_square_feet'], axis = 1)  # dropping land_square_feet 
data = data.dropna(subset=['gross_square_feet'])  # dropping NA values rows from gross_square_feet column

# %%
# MODEL BUILDING
# Linear Regression
from statsmodels.formula.api import ols

form = "SALE PRICE ~ BOROUGH + NEIGHBORHOOD + BUILDING CLASS CATEGORY + TAX CLASS AT PRESENT + BLOCK + LOT + BUILDING CLASS AT PRESENT + ZIP CODE + RESIDENTIAL UNITS + COMMERCIAL UNITS"
# modelPrice = ols(formula=form, data=data) # Syntax error

data["SALE_PRICE"] = data["SALE PRICE"]
data_sold = data[data["SALE_PRICE"]!=' -  ']
# modelPrice = ols(formula = "SALE_PRICE ~ BOROUGH", data=data) - not working
# modelPrice.summary()
# %%
# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# %%
