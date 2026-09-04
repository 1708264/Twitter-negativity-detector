import time
from funcions import loadDB, repartirDataset, dividirText, NaiveBayes  

start_time = time.perf_counter()
dataset = loadDB()
stop_time = time.perf_counter()
print("Temps d'execució loadBD:", stop_time - start_time)

train, test = repartirDataset(dataset, 0.5)

for desviacio in range(0, 10):
    desviacio /= 100
    print("\n-------------------\ndesviacio:", desviacio, "\n-------------------")

    dict, numPos, numNeg = dividirText(train, 0, 1, desviacio)
    print("Nombre de paraules després de dividirText:", len(dict))

    start_time = time.perf_counter()
    TP, TN, FP, FN = NaiveBayes(dict, test, numPos, numNeg)
    stop_time = time.perf_counter()
    print("Temps d'execució NaiveBayes:", stop_time - start_time)

    print("-------------------")
    print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)