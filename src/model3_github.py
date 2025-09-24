import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


# Manipolazione Dataset
data = pd.read_csv('3_bitcoin_clean.csv')
data['Close'] = data['Close'].interpolate(method='linear')

data['Long_Term_Price'] = data['Close'].ewm(span=288000, adjust=False).mean()
data['Short_Term_Price'] = data['Close'].ewm(span=72000, adjust=False).mean()
data['signal'] = (data['Short_Term_Price'] > data['Long_Term_Price']).astype(int)

plt.plot(data['Timestamp'], data['Close'], label='Long_Term_Price')
plt.plot(data['Timestamp'], data['Long_Term_Price'], color='orange', label='Long_Term_Price')
plt.plot(data['Timestamp'], data['Short_Term_Price'],color='red', label='Short_Term_Price')

# Training
X = data[['Short_Term_Price', 'Long_Term_Price']]
y = data['signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X_train, y_train)

y_pred = model3.predict(X_test)
yy=model3.predict(X)

data['Pred']= yy
# Valutazione modello
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
scores = cross_val_score(model3, X, y, cv=5)
std_score = np.std(scores)
print("Standard Deviation:", std_score)
print(scores)
print(scores.mean())
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

accuracytot=accuracy_score(y,yy)
print(f'Accuracy: {accuracytot}')



# Auc-roc
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Liste per memorizzare le metriche e le curve
confusion_matrices = []
roc_curves = []
roc_aucs = []

# Colori per le curve ROC
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Iterare attraverso i folds
for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Addestrare il modello
    model3.fit(X_train, y_train)
    
    # Fare previsioni sul fold di validazione
    y_val_pred = model3.predict(X_test)
    y_val_prob = model3.predict_proba(X_test)[:, 1]
    
    # Calcolare la matrice di confusione
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    confusion_matrices.append(conf_matrix)
    
    # Calcolare la curva ROC
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    roc_curves.append((fpr, tpr))
    
    # Calcolare l'area sotto la curva ROC
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)


# Model 3 on next dataset 
data1 = pd.read_csv('4_bitcoin_clean.csv')
data1['Close'] = data1['Close'].interpolate(method='linear')

data1['Long_Term_Price'] = data1['Close'].ewm(span=288000, adjust=False).mean()
data1['Short_Term_Price'] = data1['Close'].ewm(span=72000, adjust=False).mean()
data1['signal'] = (data1['Short_Term_Price'] > data1['Long_Term_Price']).astype(int)

X1 = data1[['Short_Term_Price', 'Long_Term_Price']]
y1 = data1['signal']

#Model use
data1['Pred']=model3.predict(X1)



# Signal visualization
buy=data1[data1['Pred'] == 1]
no_buy= data1[data1['Pred'] == 0]
buyp= data1[data1['signal'] == 1]
no_buyp= data1[data1['signal'] == 0]

plt.plot(data['Timestamp'], data['Close'], label='Long_Term_Price')
plt.scatter(buy.index, buy['Close'], label='Buy Signal', color='green', marker='^', alpha=0.4, s=6)
plt.scatter(no_buy.index, no_buy['Close'], label='No Buy Signal', color='red', marker='v', alpha=0.8, s=10)

plt.scatter(buyp.index, buyp['Close'], label='Buy Signal', color='green', marker='^', alpha=0.4, s=6)
plt.scatter(no_buyp.index, no_buyp['Close'], label='No Buy Signal', color='red', marker='v', alpha=0.8, s=10)

