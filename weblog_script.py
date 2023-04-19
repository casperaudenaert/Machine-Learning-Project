#importeren van de nodige libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

weblogs = pd.read_csv('weblogs.csv') # Het inlezen van de CSV file als een pandaframe

#Het verwijderen van onnodige data
weblogs = weblogs.drop('ID', axis=1)

weblogs.drop_duplicates()#Alle dubbele waarden verwijderen uit de dataframe

#De data het NaN waarden, de plaatsen waar dit voorkomt krijg de mediaan als waarde (Omdat er te veel waarden zijn)
median = weblogs["STANDARD_DEVIATION"].median()
weblogs["STANDARD_DEVIATION"].fillna(median, inplace=True)
median = weblogs["SF_FILETYPE"].median()
weblogs["SF_FILETYPE"].fillna(median, inplace=True)
median = weblogs["SF_REFERRER"].median()
weblogs["SF_REFERRER"].fillna(median, inplace=True)

#splits de dataframe in twee groepen
X = weblogs.iloc[:, :-1].values # Features
Y = weblogs.iloc[:, -1].values # Labels (Target)

#Splits de dataset in een training voor de features en de labels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

#Vraagt de input aan de gebruiker als een string
while True:
    try:
        input_str = input("Geef een rij van data in: ")
        waarden = input_str.split('\t')[1:]
        data = [int(waarden[0])] + [float(val) for val in waarden[1:]]
        break
    except ValueError:
        print("Fout: Ongeldige invoer. Probeer opnieuw.") 

if os.path.exists('model.joblib'):# Controleert dat het model al bestaat, als het al bestaat gaat het verder in de if anders word het model aangemaakt.
    csv_mtime = os.path.getmtime('weblogs.csv')# controleert wanneer het csv bestand is aangepast.
    model_mtime = os.path.getmtime('model.joblib')# controleert wanneer het model is aangepast.
    if csv_mtime > model_mtime:# Als het csv bestand aangepast is na het maken van het model wordt het model opnieuw getrained.
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=1)
        joblib.dump(model, 'model.joblib')# Slaat het model op.
else:
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=1)
    joblib.dump(model, 'model.joblib')# slaat het model op.

model = joblib.load('model.joblib')# laad het model in.
model.fit(X, Y) #Fit het model op de hele dataset

probability = model.predict_proba([data]) * 100 #Berekent de percentages
Y_pred = model.predict(X_test) # Maak voorspellingen op de test data
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Berekende Waarde')
plt.ylabel('Actuele Waarde')
plt.show()

#Print de percentages af
print('Zekerheid Mens:',format(probability[0][0], ".2f"), '%')
print('Zekerheid Robot:',format(probability[0][1], ".2f"), '%')