import time
from funcions import loadDB, repartirDataset, dividirText, NaiveBayesLpSm

start_time = time.perf_counter()
dataset = loadDB()
stop_time = time.perf_counter()
print("Temps d'execució loadBD:", stop_time - start_time)

for ratio in range(50, 100, 5):
    ratio /= 100
    print("\n-------------------\nRatio:", ratio, "\n-------------------")
    train, test = repartirDataset(dataset, ratio)

    start_time = time.perf_counter()
    dict, numPos, numNeg = dividirText(train, 0, 1, 0)
    stop_time = time.perf_counter()
    print("Nombre de paraules després de dividirText:", len(dict))
    print("Temps d'execució dividirText:", stop_time - start_time)

    start_time = time.perf_counter()
    TP, TN, FP, FN = NaiveBayesLpSm(dict, test, numPos, numNeg)
    stop_time = time.perf_counter()
    print("Temps d'execució NaiveBayes:", stop_time - start_time)

    print("-------------------")
    print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)

train, test = repartirDataset(dataset, 0.5)
for minRepeticions in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]:
    print("\n-------------------\nminRepeticions:", minRepeticions, "\n-------------------")

    dict, numPos, numNeg = dividirText(train, minRepeticions, 1, 0)
    print("Nombre de paraules després de dividirText:", len(dict))

    start_time = time.perf_counter()
    TP, TN, FP, FN = NaiveBayesLpSm(dict, test, numPos, numNeg)
    stop_time = time.perf_counter()
    print("Temps d'execució NaiveBayes:", stop_time - start_time)

    print("-------------------")
    print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)

for maxPercentatge in [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.7]:
    print("\n-------------------\nmaxPercentatge:", maxPercentatge, "\n-------------------")

    dict, numPos, numNeg = dividirText(train, 0, maxPercentatge, 0)
    print("Nombre de paraules després de dividirText:", len(dict))

    start_time = time.perf_counter()
    TP, TN, FP, FN = NaiveBayesLpSm(dict, test, numPos, numNeg)
    stop_time = time.perf_counter()
    print("Temps d'execució NaiveBayes:", stop_time - start_time)

    print("-------------------")
    print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)

for desviacio in range(0, 10):
    desviacio /= 100
    print("\n-------------------\ndesviacio:", desviacio, "\n-------------------")

    dict, numPos, numNeg = dividirText(train, 0, 1, desviacio)
    print("Nombre de paraules després de dividirText:", len(dict))

    start_time = time.perf_counter()
    TP, TN, FP, FN = NaiveBayesLpSm(dict, test, numPos, numNeg)
    stop_time = time.perf_counter()
    print("Temps d'execució NaiveBayes:", stop_time - start_time)

    print("-------------------")
    print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)

train, test = repartirDataset(dataset, 0.9)
dict, numPos, numNeg = dividirText(train, 0, 1, 0)
for ratioTest in range(10):
    ratioTest /= 10
    print("\n-------------------\nRatioTest:", 1 - ratioTest, "\n-------------------")
    train, test = repartirDataset(dataset, ratioTest)

    print("Nombre de paraules després de dividirText:", len(dict))

    start_time = time.perf_counter()
    TP, TN, FP, FN = NaiveBayesLpSm(dict, test, numPos, numNeg)
    stop_time = time.perf_counter()
    print("Temps d'execució NaiveBayes:", stop_time - start_time)

    print("-------------------")
    print("Acuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)

print("\n-------------------\nFINAL\n-------------------")