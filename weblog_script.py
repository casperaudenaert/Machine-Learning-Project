#importeren van de nodige libraries
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
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

pipeline = Pipeline([ #Pipline met verschillende stappen
    ('imputer', SimpleImputer(strategy='median')), #Imputer om de null waarden op te vullen met het gemiddelde
    ('scaler', StandardScaler()), #Veranderd de schaal van de getallen
    ('classifier', DecisionTreeClassifier(random_state=1)) #Het model dat word gebruikt; DecisionTreeClassifier
]) 

parameters = { # Mogelijke parameterr voor de pipeline (Alleen voor de classifier in dit script)
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)# Er word een grid search gedaan met de parameters en de pipeline en deze word dan in een variable geplaatst.
grid_search.fit(X_train, Y_train) #Fit de gridsearch op de hele dataset

if os.path.exists('model.joblib'):# Controleert dat het model al bestaat, als het al bestaat gaat het verder in de if anders word het model aangemaakt.
    csv_mtime = os.path.getmtime('weblogs.csv')# controleert wanneer het csv bestand is aangepast.
    model_mtime = os.path.getmtime('model.joblib')# controleert wanneer het model is aangepast.
    if csv_mtime > model_mtime:# Als het csv bestand aangepast is na het maken van het model wordt het model opnieuw getrained.
        joblib.dump(grid_search.best_estimator_, 'model.joblib')# Slaat het model op.
else:
    joblib.dump(grid_search.best_estimator_, 'model.joblib')# slaat het model op.

model = joblib.load('model.joblib')# laad het model in.

weblogs_test = pd.read_csv('weblogs_test.csv') # Het inlezen van de CSV file als een pandaframe
robot = 0
mens = 0
if 'ROBOT' in weblogs_test:
    weblogs_test = weblogs_test.drop('ROBOT', axis=1)
if 'ID' in weblogs_test:
    weblogs_test = weblogs_test.drop('ID', axis=1)
for i, row in weblogs_test.iterrows():#Overloopt elke rij, met de index ervoor
    data = row.values#Neemt de values van elke rij, behalve de eerste waarde 'ID' en de laatse waarde 'ROBOT'
    probability = model.predict_proba([data]) * 100 #Berekent de percentages
    if probability[0][0] > probability[0][1]:
        print(f'Rij {i+1}, Klasse: Mens')
    else:
        print(f'Rij {i+1}, Klasse: Robot')
    print('Zekerheid Mens:',format(probability[0][0], ".2f"), '%','Zekerheid Robot:',format(probability[0][1], ".2f"), '%')#Print de percentages af