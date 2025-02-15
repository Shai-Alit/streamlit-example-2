
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost
import pickle

pickle_loc = '../model/'
data_loc = '/workspaces/mainstorage/data/autoloan.csv'

df = pd.read_csv(data_loc)

#drop unneeded columns
df.drop(['APPDATE','CITY','STATE','DIVISION','REGION'],axis=1,inplace=True)

#drop rows with missing data
df = df.dropna()

#logSalary is the response variable
y = df['BAD']

#remove uneeded predictor variables
x_temp = df.drop(['BAD'],axis=1)

categorical_cols = x_temp.select_dtypes(include=["object"]).columns.tolist()
x_cat = x_temp[categorical_cols]


#get numeric variables
x_num = x_temp.drop(categorical_cols,axis=1)

#impute missing values
impute_values = x_num.mean()

x_num.fillna(impute_values,inplace=True)

df['REASON'].replace(np.NaN,'Other',regex = True, inplace=True)
df['JOB'].replace(np.nan,'Other',regex = True, inplace=True)

#create one-hot encoder
ohe = OneHotEncoder(sparse_output = False,handle_unknown='ignore')

#fit
ohe.fit(x_cat)

#encode categorical variables
x_cat_enc = ohe.transform(x_cat)

#convert to data frame
df_x_cat_enc = pd.DataFrame(x_cat_enc,columns=list(ohe.get_feature_names_out()))

#standardize numeric
scaler = StandardScaler()
scaled_cols=scaler.fit_transform(x_num)

#return to data frame
df_x_scaled = pd.DataFrame(scaled_cols, columns=x_num.columns)

#combine categorical and one-hot-encoded into one data frame
df_final = pd.concat([df_x_scaled,df_x_cat_enc],axis=1)

#split data into train and test
x_train, x_test, y_train, y_test = train_test_split(df_final, y, test_size=0.20,random_state=0)

#create DMatrix for faster processing
xgb_train = xgboost.DMatrix(x_train, y_train)
xgb_test = xgboost.DMatrix(x_test, y_test)

#set up training parameters
n=100
params = {
    'max_depth': 10,
    'learning_rate': 0.1,
}

#train the model
model = xgboost.train(params=params,dtrain=xgb_train,num_boost_round=n)

#make predictions and check accuracy
preds = model.predict(xgb_test)
accuracy= r2_score(y_test,preds)
print('Accuracy of the model is:', accuracy*100)

#pickle the imputation
pickle.dump(impute_values, open(pickle_loc + '/imputer.pkl', 'wb'))

#pickle the scaler
pickle.dump(scaler, open(pickle_loc + '/scaler.pkl', 'wb'))

#pickle the encoder
pickle.dump(ohe, open(pickle_loc + '/encoder.pkl', 'wb'))

#pickle the model
pickle.dump(model, open(pickle_loc + '/XGBmodel.pickle', 'wb'))