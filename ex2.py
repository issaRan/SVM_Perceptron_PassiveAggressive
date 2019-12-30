import csv
import random
from sys import argv

import numpy as np

epochs_per = 50
epochs_PA = 50
epochs_SVM = 35

# replace our first
def determine_gender(gender):
    if gender == "F":
        return [1]
    elif gender == "M":
        return [0]
    elif gender == "I":
        return [2]
    else:
        return [0, 0, 0]

# load our trainingSet data
def loadTrainingSet(num):
    # load data
    training_set = open(argv[num], "r+")
    csv_file = csv.reader(training_set)
    # replace our char to numeric.
    train_np = np.asarray([determine_gender(line[0]) + line[1:] for line in csv_file]).astype(float)
    training_set.close()
    return train_np

# load our labels.
def loadLabels():
    return np.loadtxt(argv[2])

# shuffle the data.
def makeShuffle(train_x, labels):
    zip_train_labels = list(zip(train_x, labels))
    random.shuffle(zip_train_labels)
    new_train_x, new_train_y = zip(*zip_train_labels)
    return new_train_x, new_train_y


def Perceptron_Train(train_x, labels):
    eta = 0.01
    weights = np.zeros((3, len(train_x[0])))
    for i in range(epochs_per):
        for x, y in zip(train_x, labels):
            y_hat = np.argmax(np.dot(weights, x))
            if y != y_hat:
                weights[int(y)] = weights[int(y)] + eta * x
                weights[y_hat] = weights[y_hat] - eta * x
                eta *= (1 - i / epochs_per)
    return weights


def SVM_Train(train_x, labels):
    eta = 0.01
    c = 0.1
    weights = np.zeros((3, len(train_x[0])))
    for i in range(epochs_SVM):
        for x, y in zip(train_x, labels):
            y_hat = np.argmax(np.dot(weights, x))
            # update
            if y != y_hat:
                weights[int(y), :] = (1 - eta * c) * weights[int(y)] + eta * x
                weights[y_hat, :] = (1 - eta * c) * weights[y_hat, :] - eta * x
                eta *= (1 - i / epochs_per)
    return weights


def tau_func(w_of_y, w_of_y_hat, x):
    norm = np.linalg.norm(x)
    if norm == 0:
        norm = 1
    loss = np.maximum(0, 1 - np.dot(w_of_y, x) + np.dot(w_of_y_hat, x))
    return loss / (2 * (norm ** 2))


def PA_Train(train_x, labels):
    weights = np.zeros((3, len(train_x[0])))
    for i in range(epochs_PA):
        for x, y in zip(train_x, labels):
            y_hat = np.argmax(np.dot(weights, x))
            if y != y_hat:
                tau = tau_func(weights[int(y), :], weights[int(y_hat), :], x)
                tx = tau * x * (1 - i / epochs_PA) * 0.4
                weights[int(y), :] += tx
                weights[int(y_hat), :] -= tx
    return weights


def error_rate(w, test_x, test_y, algo):
    bad = 0
    for i, line in enumerate(test_x):
        y_hat = np.argmax(np.dot(w, line))
        if y_hat != test_y[i]:
            bad += 1
    return 100 - (bad / len(test_y) * 100)

def prediction(weights_per, weights_svm, weights_pa, test_x):
    for classify in test_x:
        y_hat_per = np.argmax(np.dot(weights_per, classify))
        y_hat_svm = np.argmax(np.dot(weights_svm, classify))
        y_hat_pa = np.argmax(np.dot(weights_pa, classify))
        print(f"perceptron: {y_hat_per}, svm: {y_hat_svm}, pa: {y_hat_pa}")


def normal_data(data):
    columns = 8
    min = np.zeros(columns)
    max = np.zeros(columns)
    c_data = np.array(data)
    for j in range(columns):
        min[j] = np.min(c_data[:, j])
        max[j] = np.max(c_data[:, j])
    normalized = np.copy(data)
    for i, line in enumerate(normalized):
        for j in range(columns):
            if max[j] == min[j]:
                normalized[i][j] = data[i][j] - min[j]
            else:
                normalized[i][j] = (data[i][j] - min[j]) / (max[j] - min[j])
    return normalized


def loadLabelsTest():
    return np.loadtxt(argv[4])


def main(argv):
    train_x = loadTrainingSet(1)
    labels = loadLabels()
    test_x = loadTrainingSet(3)
    train_x = normal_data(train_x)
    train_x, labels = makeShuffle(train_x, labels)
    w_Perceptron = Perceptron_Train(train_x, labels)
    w_SVM = SVM_Train(train_x, labels)
    w_PassiveAggressive = PA_Train(train_x, labels)
    prediction(w_Perceptron, w_SVM, w_PassiveAggressive)


if __name__ == '__main__':
    main(argv)
