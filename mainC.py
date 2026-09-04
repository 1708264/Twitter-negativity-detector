import time
from funcions import loadDB, repartirDataset, dividirText, NaiveBayes

start_time = time.perf_counter()
dataset = loadDB()
stop_time = time.perf_counter()
print("Temps d'execució loadBD:", stop_time - start_time)

start_time = time.perf_counter()
train, test = repartirDataset(dataset, 0.5)
stop_time = time.perf_counter()
print("Temps d'execució repartirDataset:", stop_time - start_time)

start_time = time.perf_counter()
dict, numPos, numNeg = dividirText(train, 0, 1, 0)
stop_time = time.perf_counter()
print("Nombre de paraules després de dividirText:", len(dict))
print("Temps d'execució dividirText:", stop_time - start_time)

start_time = time.perf_counter()
TP, TN, FP, FN = NaiveBayes(dict, test, numPos, numNeg)
stop_time = time.perf_counter()
print("Temps d'execució NaiveBayes:", stop_time - start_time)

print("-------------------")
print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)
