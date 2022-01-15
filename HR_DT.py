import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\10.Decision Tree\HR_DT.csv')

df.isnull().sum()

X = df.drop(' monthly income of employee', axis = 1)
Y = df[' monthly income of employee']

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

X['Position of the employee'] = label.fit_transform(X['Position of the employee'])

X1

from sklearn.model_selection import train_test_split

x, x_test, y , y_test = train_test_split(X,Y, test_size = 0.2 )

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5)

model.fit(x,y)

model.score(x,y) #train accuracy

pred_test = model.predict(x_test)
pred_train = model.predict(x)

model.score(x_test, y_test) #test accuracy

from sklearn.metrics import mean_squared_error


mean_squared_error(y_test, pred_test) # test error 6867029 -MSE
mean_squared_error(y, pred_train) #train error  4203488 - MSE

np.sqrt(mean_squared_error(y_test, pred_test))  #test error 2620.5 -RMSE
np.sqrt(mean_squared_error(y, pred_train)) #train error  2050.2 - RMSE

import seaborn as sns

#plotting predicted and actual vaLues 

sns.set()    
sns.lineplot(y = pred_test , x = range(1,41) ,label = 'predicted values')
sns.lineplot(y = y_test, x = range(1,41) ,label = 'Actual values')


from sklearn.ensemble import RandomForestRegressor

model_RF = RandomForestRegressor(n_estimators=10, max_depth=3)

model_RF.fit(x,y)

model.score(x,y) #train accuracy

pred_RF = model_RF.predict(x_test)

pred_train_RF = model_RF.predict(x)

model.score(x_test,y_test) #train accuracy


mean_squared_error(y_test, pred_RF) # test error 8159871.87 -MSE
mean_squared_error(y, pred_train_RF) #train error  9119723.73 - MSE

np.sqrt(mean_squared_error(y_test, pred_RF))  #test error 2856.54 -RMSE
np.sqrt(mean_squared_error(y, pred_train_RF)) #train error  3019.8 - RMSE


