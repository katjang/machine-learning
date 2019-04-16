"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)


STUDENTNUMMER = "0884141"

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

#print(X)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# teken de punten
for i in range(len(x)):
    plt.plot(x[i], y[i], 'k.') # k = zwart

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='Set3')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.axis([min(x), max(x), min(y), max(y)])
plt.show()




# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)
Y = extract_from_json_as_np_array("y", classification_training)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

clf = GaussianNB()
clf.fit(X, Y)

model = LogisticRegression()
model.fit(X, Y)

decision = DecisionTreeClassifier()
decision.fit(X, Y)

GausPredict = clf.predict(X)
logisticPredict = model.predict(X)
decisionPredict = decision.predict(X)

print("GaussianNB Test Accuracy rating: " + str(accuracy_score(Y, GausPredict)))
print("LogisticRegression Test Accuracy rating: " + str(accuracy_score(Y, logisticPredict)))
print("DecisionTreeClassifier Test Accuracy rating: " + str(accuracy_score(Y, decisionPredict)))

# TODO: leer de classificaties, en kijk hoe goed je dat gedaan hebt door je voorspelling te vergelijken
# TODO: met de werkelijke waarden


# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)



pred = clf.predict(X_test)

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(pred.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie Test accuratie (Gaussian): " + str(classification_test))

pred = model.predict(X_test)

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(pred.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie Test accuratie (LogisticRegression): " + str(classification_test))

pred = decision.predict(X_test)

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(pred.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie Test accuratie (DecisionTreeClassifier): " + str(classification_test))



