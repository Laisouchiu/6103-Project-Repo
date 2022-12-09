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
data['TAX CLASS AT PRESENT'].value_counts()
# %%
class_sale_count = data['BUILDING CLASS AT TIME OF SALE'].value_counts()
class_sale_count
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
# %%
# DATA VISUALIZATION



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
