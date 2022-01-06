# Libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset
df= pd.read_csv("Life Expectancy Data.csv")

df.columns= [i.strip() for i in df.columns]

# Artiq hec bir olkenin hepatiti sutunu tamamen bos deyil (Yuxarikina gore)
hep_bos= df.groupby(by="Country")['Hepatitis B'].mean()
ind_hep= df[df['Hepatitis B'].isnull()]['Hepatitis B'].index

sil=hep_bos[hep_bos.isnull()].index
stable_countries=list(set(df['Country'].unique()).difference(set(sil)))
df=(df[((df['Country'].isin(stable_countries)))])

# Kicik bir xeta var hell edek
df=df[df['Country']!=75.1875]

df= df[df['Population'].notnull()]
df["Hepatitis B"]=df['Hepatitis B'].fillna(method="bfill")
df=df[df['Alcohol'].notnull()]


# EDA Bitdi
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

num_cols= [i  for i in df.select_dtypes(include=['int64','float64']).columns if i!='Life expectancy']
cat_cols= [i for i in df.select_dtypes(include=['O','object'])]

# Pipelines
num_pip= Pipeline(steps=[
    ("imputen",SimpleImputer(strategy='mean')),
    ('scale',MinMaxScaler())
])

cat_pip= Pipeline(steps=[
    ("imputec",SimpleImputer(strategy='most_frequent')),
    ("encode",OneHotEncoder())
])

col_tr= ColumnTransformer(transformers=[
    ("Numeric",num_pip,num_cols),
    ("Category",cat_pip,cat_cols)
])

tree_pip= Pipeline(steps=[
    ("Prep",col_tr),
    ("model",DecisionTreeRegressor())
])

df=df[df['Country']!="Tuvalu"]

x=df.drop(columns='Life expectancy')
y=df['Life expectancy']
x_train,x_test,y_train,y_tets= train_test_split(x,y)


knn_pip= Pipeline(steps=[
    ("Prep",col_tr),
    ("model",KNeighborsRegressor(weights='distance'))
])

line_pip= Pipeline(steps=[
    ("Prep",col_tr),
    ("model",LinearRegression())
])


svr_pip= Pipeline(steps=[
    ("Prep",col_tr),
    ("model",SVR())
])

voting_model= VotingRegressor(estimators=[
    ("tree",tree_pip),
    ("knn",knn_pip),
    ("linear",line_pip),
    ("svr",svr_pip)
])

voting_model.fit(x_train,y_train)

# pickle-ye yuklemek

pickle.dump(voting_model,open("model.pkl","wb"))
