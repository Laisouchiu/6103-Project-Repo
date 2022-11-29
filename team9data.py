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
data = pd.read_csv("Team9 data.csv")
data.head()
# %%
data.shape # 84,548 rows and 22 columns
# %%
data.describe()
# %%
