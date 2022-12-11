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
#%%
data.isna().sum()
# %%
data.info()
# %%
data = data.drop("Unnamed: 0", axis=1)
# %%
newcols = ["borough", "neighborhood", "building_class_category", "tax_class_at_present", "block", "lot", "easement", "building_class_at_present", "address", "apartment_number", "zip_code", "residential_units", "commercial_units", "total_units", "land_square_feet", "gross_square_feet", "year_built", "tax_class_at_time_of_sale", "building_class_at_time_of_sale", "sale_price", "sale_date"]
data.columns = newcols
data.columns
# %%
# converting sale_date to datetime datatype
data['sale_date'] = pd.to_datetime(data['sale_date'])

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
round(data.isna().sum() /len(data) *100,2)
# %%
data = data.drop(['easement'], axis = 1)
# %%
print(data[(data['land_square_feet'].isnull()) & (data['gross_square_feet'].notnull())].shape[0])
print(data[(data['land_square_feet'].notnull()) & (data['gross_square_feet'].isnull())].shape[0])

# This indicate we have 1372 values can be imputed
# %%
fillValues = data[['land_square_feet','gross_square_feet']]
len(fillValues[(fillValues['land_square_feet'].isnull()) & (fillValues['gross_square_feet'].notnull())])
# %%
fillValues.dropna(inplace = True)
# %%
from sklearn.linear_model import LinearRegression

X = fillValues['land_square_feet']
y = fillValues['gross_square_feet']

# model to predict gross_square_feet
lm = LinearRegression()
model_to_find_gross_sqft = lm.fit(X.values.reshape(-1,1),y.values.reshape(-1,1))

# %%
y = fillValues['land_square_feet']
X = fillValues['gross_square_feet']

# model to predict land_square_feet
lm = LinearRegression()
model_to_find_land_sqft = lm.fit(X.values.reshape(-1,1),y.values.reshape(-1,1))
# %%
print(f"Eqn to find gross_sqft: y = {model_to_find_gross_sqft.coef_}x + {model_to_find_gross_sqft.intercept_}")
print(f"Eqn to find land_sqft: y = {model_to_find_land_sqft.coef_}x + {model_to_find_land_sqft.intercept_}")
# %%
def find_gross_sqft(x):
    return x * 0.64815423 + 1701.2484797

# land square feet estimator

def find_land_sqft(x):
    return x * 0.63778678 + 1033.76397579
# %%
data['land_square_feet'] = data['land_square_feet'].mask((data['land_square_feet'].isnull()) & (data['gross_square_feet'].notnull()),
                                                     find_land_sqft(data['gross_square_feet']))
data['gross_square_feet'] = data['gross_square_feet'].mask((data['land_square_feet'].notnull()) & (data['gross_square_feet'].isnull()),
                                                       find_gross_sqft(data['land_square_feet']))
# %%
data = data[data['sale_price']>0]
data = data[data["sale_price"].notnull()] 
# %%
data = data.drop(['apartment_number'], axis=1)
# %%
# Removing data where commercial + residential doesn't equal total units
data = data[data['total_units'] == data['commercial_units'] + data['residential_units']]
# %%
data = data.drop(['address'], axis=1)
# %%
data = data.dropna()
# %%
data.drop_duplicates(keep = "last", inplace=True)
# %%
data[["total_units", "sale_price"]].groupby(['total_units'], as_index=False).count().sort_values(by='sale_price', ascending=False)
# %%
# Removing rows with TOTAL UNITS == 0 and one outlier with 2261 units
data = data[(data['total_units'] > 0) & (data['total_units'] != 2261)]
# %%
data['year_built'].sort_values()
# %%
data = data[data['year_built'] != 0]
data['age'] = 2022 - data['year_built']
# %%
data = data.drop(['year_built'], axis =1)
# %%
data = data[data["land_square_feet"] != 0]
data = data[data["gross_square_feet"] != 0]
# %%
len(data)
# %%
data.shape
# %%
plt.boxplot(data['land_square_feet'])
# %%
sns.boxplot(data = data, x='gross_square_feet')

#%%
round(data.describe([0.75,0.95,0.99]),3)
# %%
# removing few outliers
data = data[data['land_square_feet'] < 72000]
data = data[data['gross_square_feet'] < 70000]
# %%
data.corr()

# %%
ax = sns.heatmap(data.corr(), annot=True)
plt.title("Correlation matrix")
# %%
# from the correlation matrix, very less correlation of sales price on block, lot
# so dropping block and lot

data = data.drop(['block', 'lot'], axis=1)
# %%
# some visualizations
df2 = data[(data['sale_price'] > 100000) & (data['sale_price'] < 5000000)]

sns.boxplot(x='borough', y='sale_price', data=df2, palette='rainbow')
# %%
sns.barplot(x='borough', y='land_square_feet', data=data)
# %%
sns.barplot(x='borough', y='gross_square_feet', data=data)
# %%
