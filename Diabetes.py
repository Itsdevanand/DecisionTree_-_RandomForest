import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\10.Decision Tree\Diabetes.csv')

df.head()
df.isnull().sum() #No na values


X = df.drop(' Class variable', axis = 1) #input

Y = df[' Class variable']  #output


#Normalising the data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

print(scaler.fit(X))
X1 = pd.DataFrame(scaler.fit_transform(X))
X1.columns = X.columns
X = X1

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
Y = label.fit_transform(Y)

from sklearn.model_selection import train_test_split

x, x_test, y , y_test = train_test_split(X,Y, test_size=0.2)

from sklearn import tree

model1 = tree.DecisionTreeClassifier()

model1.fit(x,y)

model1.score(x, y)

pred_DT = model1.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred_DT)

#plotting DT

plt.figure(figsize=(115,30) )
tree.plot_tree(model1, filled=True)

#We can see that it is cleary overfitting so we use pruning to reduce overfitting
#Using Cost complexity pruning fro getting proper depth of the tree

prun  = model1.cost_complexity_pruning_path(x,y)

ccp_values = prun['ccp_alphas']

test_acc = []
train_acc = []

for i in ccp_values:
    
    model2 = tree.DecisionTreeClassifier(ccp_alpha = i  )
    model2.fit(x,y)
    train_acc.append(model2.score(x,y))
    pred = model2.predict(x_test)
    test_acc.append(accuracy_score(y_test, pred))
    
#plotting two accuracy values

import seaborn as sns

sns.set()    
sns.lineplot(y = test_acc, x = ccp_values, label = 'Test Accuracy')
sns.lineplot(y = train_acc, x = ccp_values, label = 'Train accuracy')

#from the plot, the ccp_alpha value of 0.005

#final model

model_DT = tree.DecisionTreeClassifier(ccp_alpha = 0.005)

model_DT.fit(x,y)

model_DT.score(x, y)  #Train accuracy = 0.80
pred_DT = model_DT.predict(x_test)
accuracy_score(y_test, pred_DT) #Test accuracy  = 0.76


