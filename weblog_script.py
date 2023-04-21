#importeren van de nodige libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

weblogs = pd.read_csv('weblogs.csv') # Het inlezen van de CSV file als een pandaframe

#Het verwijderen van onnodige data
weblogs = weblogs.drop('ID', axis=1)

weblogs.drop_duplicates()#Alle dubbele waarden verwijderen uit de dataframe

#splits de dataframe in twee groepen
X = weblogs.iloc[:, :-1].values # Features
y = weblogs.iloc[:, -1].values # Labels (Target)

#Splits de dataset in een training voor de features en de labels
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=1)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=1))
])

parameters = {
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, Y_train)

if os.path.exists('model.joblib'):# Controleert dat het model al bestaat, als het al bestaat gaat het verder in de if anders word het model aangemaakt.
    csv_mtime = os.path.getmtime('weblogs.csv')# controleert wanneer het csv bestand is aangepast.
    model_mtime = os.path.getmtime('model.joblib')# controleert wanneer het model is aangepast.
    if csv_mtime > model_mtime:# Als het csv bestand aangepast is na het maken van het model wordt het model opnieuw getrained.
        #model = DecisionTreeClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=1)
        joblib.dump(grid_search.best_estimator_, 'model.joblib')# Slaat het model op.
else:
    #model = DecisionTreeClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=1)
    joblib.dump(grid_search.best_estimator_, 'model.joblib')# slaat het model op.

model = joblib.load('model.joblib')# laad het model in.


weblogs_test = pd.read_csv('weblogs_test.csv') # Het inlezen van de CSV file als een pandaframe
#Het verwijderen van onnodige data
robot = 0
mens = 0
#Overloopt alle rijen van de test data en voorspelt of het een robot of een mens is
for i, row in weblogs_test.iterrows():
    data = row.values[1:-1]
    probability = model.predict_proba([data]) * 100 #Berekent de percentages
    #Print de percentages af
    print('Zekerheid Mens:',format(probability[0][0], ".2f"), '%','Zekerheid Robot:',format(probability[0][1], ".2f"), '%')