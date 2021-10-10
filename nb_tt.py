import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB,BernoulliNB
def PCNB(train_features, train_labels):
    train_labels_set = set(train_labels)
    PC = {}
    for i in train_labels_set:
        PC['C='+str(i)] = train_labels.count(i)/float(len(train_labels))
    for i in PC
        print('P(Y='+i+')=',PC[i])
    A_set = []
    for i in range(train_features[0]):
        A_set.append(set(train_features[:,i]))
    PA = {}
    for label in train_labels_set:


df_train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
t_result=pd.read_csv('gender_submission.csv')
test.insert(0, 'Survived', t_result['Survived'])
df_train = df_train.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df_train.isnull().sum())
print(df_train.Embarked.mode())
df_train['Embarked'].fillna('S', inplace=True)
print(df_train.isnull().sum())
df_train['Sex_cleaned'] = np.where(df_train['Sex'] == 'female', 1, 0)
df_train['Embarked_cleaned'] = np.where(df_train['Embarked'] == 'S', 0, np.where(df_train['Embarked'] == 'C', 1, np.where(df_train['Embarked'] == 'Q', 2, 3)))
test['Sex_cleaned'] = np.where(test['Sex'] == 'female', 1, 0)
test['Embarked_cleaned'] = np.where(test['Embarked'] == 'S', 0, np.where(test['Embarked'] == 'C', 1, np.where(test['Embarked'] == 'Q', 2, 3)))
print(test.head())
df_train=df_train[[
    'Survived',
    'Pclass',
    'Sex_cleaned',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked_cleaned'
]].dropna(axis=0, how='any')
test=test[[
    'Survived',
    'Pclass',
    'Sex_cleaned',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked_cleaned'
]].dropna(axis=0, how='any')
gnd = GaussianNB()
titanic_features = [
    'Pclass',
    'Sex_cleaned',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked_cleaned'
]
gnd.fit(df_train[titanic_features].values,df_train['Survived'])
y_axis = gnd.predict(test[titanic_features])
print(test.shape[0], (test['Survived'] != y_axis).sum())


