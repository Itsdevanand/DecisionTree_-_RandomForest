import pandas as pd.
import numpy as np

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\10.Decision Tree\Fraud_check.csv')

df.isnull().sum()

Y = pd.DataFrame(df['Taxable.Income'])

Y['Taxable.Income'] = np.where(Y['Taxable.Income'] <= 30000 ,1,0) #Converting to discrete data

X = df.drop('Taxable.Income', axis=1)
        
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

#label encoding

X['Undergrad'] = label.fit_transform(X.Undergrad)  
X['Marital.Status'] = label.fit_transform(X['Marital.Status'])
X['Urban'] = label.fit_transform(X.Urban)

from sklearn.model_selection import train_test_split

x, x_test, y , y_test = train_test_split(X,Y, test_size = 0.2 )

from sklearn import tree

model_DT = tree.DecisionTreeClassifier()

#cost complexity pruning

prun  = model_DT.cost_complexity_pruning_path(x,y)

ccp_values = prun['ccp_alphas']

test_acc = []
train_acc = []

from sklearn.metrics import accuracy_score

for i in ccp_values:
    
    model_DT = tree.DecisionTreeClassifier(ccp_alpha = i  )
    model_DT.fit(x,y)
    train_acc.append(model_DT.score(x,y))
    pred = model_DT.predict(x_test)
    test_acc.append(accuracy_score(y_test, pred))
    
import seaborn as sns

#plotting alpha values and accuracy
sns.set()    
sns.lineplot(y = test_acc, x = ccp_values, label = 'Test Accuracy')
sns.lineplot(y = train_acc, x = ccp_values, label = 'Train accuracy')

#optimum accuracy is at 0.005

model_DT = tree.DecisionTreeClassifier(ccp_alpha = 0.005 )

model_DT.fit(x,y)

model_DT.score(x, y)

pred_DT = model_DT.predict(x)

accuracy_score(y_test, pred_DT )

from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators=100, max_depth=5 )

model_RF.fit(x,y)

model_RF.score(x,y) #train accuracy 0.81

pred_RF = model_RF.predict(x_test)

accuracy_score(y_test, pred_RF) #test accuracy = 0.75

