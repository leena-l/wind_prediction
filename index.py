import pandas as pd
import pickle
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore") 


import seaborn as sns


df=pd.read_csv('Data.csv')
df=df.dropna() 
df.info()

df.describe()
df.isnull().sum()

df.shape

col= df.columns
col

for i in col:
    plt.figure(figsize=(9, 8))
    sns.distplot(df[i], color='g', bins=100, hist_kws={'alpha': 0.4})

sns.pairplot(df)



relation= df.corr()
top_corr_features = relation.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")




cdf = df[['T','TM','Tm','SLP','H','VV','V','PM 2.5','VM']]


x = cdf.iloc[:, :8]
y = cdf.iloc[:, -1]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



from sklearn.ensemble import ExtraTreesRegressor

model2 = ExtraTreesRegressor()
model2.fit(x_train, y_train)
print(model2.feature_importances_)

feat_importances = pd.Series(model2.feature_importances_, index=x.columns)
feat_importances.plot(kind='barh')
plt.show()


linearRegression = LinearRegression()
#linearRegression.fit(x, y)
linearRegression.fit(x_train, y_train)
y_pred = linearRegression.predict([[7.4,9.8,4.8,1017.6,93.0,0.5,4.3,219.720]])
#lracc = linearRegression.score(x,y)
lracc = linearRegression.score(x_train, y_train)


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=44)
model.fit(x_train, y_train)
y_pred = model.predict([[7.4,9.8,4.8,1017.6,93.0,0.5,4.3,219.720]])
dtacc = model.score(x_train, y_train)

SVM = SVR()
SVM.fit(x_train, y_train)
y_pred = SVM.predict([[7.4,9.8,4.8,1017.6,93.0,0.5,4.3,219.720]])
SVMacc = SVM.score(x_train, y_train)
print(SVMacc)


clf_acc=round(linearRegression.score(x_train, y_train), 4)
RFacc=round(model.score(x_train, y_train), 4)
svm_acc=round(SVM.score(x_train, y_train), 4)


y_pred = model.predict(x_test)

df2 = y_test.mean()


import numpy as np
cutoff = 16                              # decide on a cutoff limit
y_pred_classes = np.zeros_like(y_pred)    # initialise a matrix full with zeros
y_pred_classes[y_pred > cutoff] = 1 



y_test_classes = np.zeros_like(y_test)
y_test_classes[y_test > cutoff] = 1

conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)





fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Precision: %.3f' % precision_score(y_test_classes, y_pred_classes))

print('Recall: %.3f' % recall_score(y_test_classes, y_pred_classes))

print('Accuracy: %.3f' % accuracy_score(y_test_classes, y_pred_classes))

print('F1 Score: %.3f' % f1_score(y_test_classes, y_pred_classes))


data = {'LogisticRegression':clf_acc, 'SVC':svm_acc, 'DecisionTreeRegressor':RFacc}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color =['black', 'red', 'green'], 
        width = 0.4)
 
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy of Algorithms")
plt.show()



file=open('my_model.pkl','wb')
pickle.dump(model,file,protocol=3)