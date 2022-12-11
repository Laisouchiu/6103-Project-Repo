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
#%%
# %%
data_sold_features = data_few_cols[["borough", "building_class_category", "zip_code", "total_units", "age", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price", "gross_square_feet", "land_square_feet", "percent_resdiential_units"]]
dataPreprocessor = ColumnTransformer( transformers=
    [
        ("categorical", OneHotEncoder(), ["building_class_category", "tax_class_at_time_of_sale", 
                                          "building_class_at_time_of_sale", "borough", "zip_code"
                                          ]),
        ("numeric", StandardScaler(), ["total_units", "age", "sale_price", "gross_square_feet", "land_square_feet", "percent_residential_units"])
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