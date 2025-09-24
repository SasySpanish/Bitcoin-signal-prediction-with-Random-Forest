import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #or whatever
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

#Data manipulation
data = pd.read_csv('')
#data[''] = data['].interpolate(method='linear')  #Interpolazione per valori mancanti (NaN)
data1=data.loc[0:3] #split dataset

# plt.plot(data['tempo'], data['var'],color='blue' label='nome') #Grafico
plt.legend()

#Training
X = data[['Short_Term_Price', 'Long_Term_Price']]
y = data['signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Applicazione modello ad un altro dataset
X1 = data1[['Short_Term_Price', 'Long_Term_Price']]
y1 = data1['signal']
predictions = model.predict(X1) #importante
data['Pred']= predictions


# Valuta il modello
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Cross validation
scores = cross_val_score(model, X, y, cv=5)
std_score = np.std(scores)
print("Cross Validation Scores", scores)
print("Standard Deviation:", std_score)
print("Mean:", scores.mean())

# Print cross validation
pscores = np.array([0.80491705, 0.61924134, 0.89466741, 0.62367006])

# Media dei punteggi
mean_score = pscores.mean()

# Deviazione standard dei punteggi
std_score = pscores.std()

# Plot dei punteggi individuali (bar plot)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pscores) + 1), pscores, color='skyblue', edgecolor='black')
plt.axhline(y=mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
plt.errorbar(range(1, len(pscores) + 1), pscores, yerr=std_score, fmt='o', color='black', label=f'Std Dev: {std_score:.2f}')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('Cross-Validation Scores')
plt.legend()


#Matrice di Confusione
conf_matrix = confusion_matrix(y1, predictions)
print("Confusion Matrix on Validation Set:")
print(conf_matrix)

# Calcolare l'AUC-ROC

val_probabilities = model.predict_proba(X1)[:, 1]  # Probabilità previste per la classe positiva
val_auc_roc = roc_auc_score(y1, val_probabilities)
print(f"Validation Set AUC-ROC: {val_auc_roc:.4f}")

fpr, tpr, thresholds = roc_curve(y1, val_probabilities)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % val_auc_roc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#Segnali 
buy=data[data['signal'] == 1]
no_buy= data[data['signal'] == 0]

plt.scatter(buy.index, buy['Close'], label='Buy Signal', color='green', marker='^', alpha=0.4, s=3)
plt.scatter(no_buy.index, no_buy['Close'], label='No Buy Signal', color='red', marker='v', alpha=0.4, s=1)
plt.legend()

buy1=data1[data1['Pred'] == 1]
no_buy1= data1[data1['Pred'] == 0]

plt.scatter(buy1.index, buy1['Close'], label='Buy Signal', color='green', marker='^', alpha=0.4, s=3)
plt.scatter(no_buy1.index, no_buy1['Close'], label='No Buy Signal', color='red', marker='v', alpha=0.4, s=1)
plt.legend()












