import random
import math


def loadDB():
    dataset = []
    for i in range(6):
        with open(f"dataset_part{i}.csv", "r") as file:
            if i == 0:
                next(file)
            for line in file:
                line = line.split(";")
                dataset.append({"message": line[1].strip().split(" "), "label": int(line[3].strip())})
    return dataset


def repartirDataset(dataset, ratio):
    positius = []
    negatius = []
    for d in dataset:
        if d["label"] == 1:
            positius.append(d)
        elif d["label"] == 0:
            negatius.append(d)
    random.shuffle(positius)
    random.shuffle(negatius)
    train = positius[:int(len(positius) * ratio)] + negatius[:int(len(negatius) * ratio)]
    test = positius[int(len(positius) * ratio):] + negatius[int(len(negatius) * ratio):]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def dividirText(train, minRepeticions, maxPercentatge, desviacio):
    proporcio = {}
    numPos = 0
    numNeg = 0
    for values in train:
        for word in values["message"]:
            if word not in proporcio:
                proporcio[word] = {0:0, 1:0}
            proporcio[word][values["label"]] += 1
            if values["label"] == 1:
                numPos += 1
            else:
                numNeg += 1
    paraulesAEliminar = []
    for word in proporcio:
        proporcio[word]["percentatge"] = proporcio[word][1] / (proporcio[word][1] + proporcio[word][0])
        if (proporcio[word][0] + proporcio[word][1]) <= minRepeticions:
            paraulesAEliminar.append(word)
        elif proporcio[word]["percentatge"] > maxPercentatge or proporcio[word]["percentatge"] < 1 - maxPercentatge:
            paraulesAEliminar.append(word)
        elif proporcio[word]["percentatge"] < 0.5 + desviacio and proporcio[word]["percentatge"] > 0.5 - desviacio:
            paraulesAEliminar.append(word)
    for paraula in paraulesAEliminar:
        del proporcio[paraula]
    return proporcio, numPos, numNeg


def NaiveBayes(train, test, numPos, numNeg):
    novesParaules = []
    priorPos = math.log(numPos / (numPos + numNeg))
    priorNeg = math.log(numNeg / (numPos + numNeg))
    TP, TN, FP, FN = 0, 0, 0, 0
    for tweet in test:
        logPos = priorPos
        logNeg = priorNeg
        for word in tweet["message"]:
            if logPos == float("-inf") and logNeg == float("-inf"):
                break
            if word in train:
                if train[word][1] > 0:
                    logPos += math.log(train[word][1] / numPos)
                else:
                    logPos += float("-inf")

                if train[word][0] > 0:
                    logNeg += math.log(train[word][0] / numNeg)
                else:
                    logNeg += float("-inf")
            else:
                novesParaules.append(word)
        if logPos > logNeg:
            prediccio = 1
        else:
            prediccio = 0
        if prediccio == tweet["label"]:
            if prediccio == 1:
                TP += 1
            else:
                TN += 1
        else:
            if prediccio == 1:
                FP += 1
            else:
                FN += 1
                    
    print("Noves paraules trobades durant la prova:", len(novesParaules))
    return TP, TN, FP, FN

def NaiveBayesLpSm(train, test, numPos, numNeg):
    novesParaules = []
    priorPos = math.log(numPos / (numPos + numNeg))
    priorNeg = math.log(numNeg / (numPos + numNeg))
    TP, TN, FP, FN = 0, 0, 0, 0
    TL = len(train)
    for tweet in test:
        logPos = priorPos
        logNeg = priorNeg
        for word in tweet["message"]:
            if word in train:
                logPos += math.log((train[word][1] + 1) / (numPos + TL))
                logNeg += math.log((train[word][0] + 1) / (numNeg + TL))
            else:
                novesParaules.append(word)
                logPos += math.log(1 / (numPos + TL))
                logNeg += math.log(1 / (numNeg + TL))
        if logPos > logNeg:
            prediccio = 1
        else:
            prediccio = 0
        if prediccio == tweet["label"]:
            if prediccio == 1:
                TP += 1
            else:
                TN += 1
        else:
            if prediccio == 1:
                FP += 1
            else:
                FN += 1
                    
    print("Noves paraules trobades durant la prova:", len(novesParaules))
    return TP, TN, FP, FN
