from sklearn import datasets
import numpy as np


iris = datasets.load_iris()
trainingSet = np.array(iris.data)
answers = np.array(iris.target)
names = np.array(iris.target_names)

INPUT_DIM = 4
OUTPUT_DIM = 3
NEURONS = 10
E = 0.7
Δw_O = np.zeros(10)
Δw_H = np.zeros(4)


def sigmoid(x):                 # activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(output):
    return (1 - output) * output

weightsH = np.array([[-19.36586277, -19.27567501,  -4.29929792,  -0.16506057,   0.61277211, 1.24013347,   1.09457793,   0.84237402,   0.76156881,   0.50596052],
                           [-18.41174343, -11.46289381,  -4.12392511,  -0.49134979,   0.42304295, 0.31459498,   0.7776716,    0.96867287,   1.05306631,   0.88855117],
                           [ 27.9008442,   18.60038451,   9.58807802,   3.9338413,    1.72069541, 0.39958331,   0.36595397,   0.54770805,   0.60770143,   1.01342532],
                           [ 23.94640285,  17.26528702,   7.13688123,   3.06379159,   0.78460879, 0.45909744,   0.59741049,   0.76379239,   0.33625582,   0.74154091]])
b1 = np.random.rand(NEURONS)
weightsO = np.array([[-0.13675626, -6.7952959, 6.72587779],
                     [-2.80427125, -5.96197919, 6.25249772],
                     [-10.37493679, 7.46809519, 7.15179387],
                     [1.06096847, -0.11595377, -1.54887123],
                     [0.668329, -0.55537728, -1.61346918],
                     [0.83714776, -1.26864966, -1.68367068],
                     [0.47452737, -1.11518132, -1.83965771],
                     [1.2119406, -0.29524062, -1.00581222],
                     [0.58545422, -0.79528677, -1.34119806],
                     [0.14996988, -0.46644062, -1.0485709]])
b2 = np.random.rand(OUTPUT_DIM)

def ai_learning():
    for i in range(1000):
        for k in range(150):

                x = np.array(trainingSet[k])    # forward propagation
                t1 = x @ weightsH
                h1 = sigmoid(t1)
                t2 = h1 @ weightsO
                outputValue = sigmoid(t2)
                                                # back propagation
                out_ideal = np.zeros(3)

                if answers[k] == 0:         # checking the answer
                    out_ideal[0] = 1
                elif answers[k] == 1:
                    out_ideal[1] = 1
                else:
                    out_ideal[2] = 1

                δO = (out_ideal - outputValue) * sigmoid_deriv(outputValue)     # delta
                δH = sigmoid_deriv(h1) * np.transpose(np.sum(weightsO * δO, axis=1))

                for j in range(3):              # gradient and new weights
                    GRAD_O = h1 * δO[j]
                    Δw_O = GRAD_O * E + Δw_O * 0.3
                    weightsO[:, j]= np.transpose(Δw_O) + weightsO[:, j]
                for j in range(10):
                    GRAD_H = trainingSet[k] * δH[j]
                    Δw_H = GRAD_H * E + Δw_H * 0.3
                    weightsH[:, j] = np.transpose(Δw_H) + weightsH[:, j]
                print(weightsH)


def output(x):         # forward propagation
    t1 = x @ weightsH
    h1 = sigmoid(t1)
    t2 = h1 @ weightsO
    return sigmoid(t2)


x = np.zeros(4)     # input vector

print("-----------------------------------------------",
      "////////////////APE.INC__S1MORI////////////////",
      "-----------------------------------------------", "\n",
      "                 Fisher Iris                   ",
      "choose option:                                 ",
      "1: classify   2: retrain the network           ",
      "input: ", sep="\n")

option = int(input())

if option == 1:
    print("-----------------------------------------------", "\n",
          "////////////////APE.INC__S1MORI////////////////", "\n",
          "-----------------------------------------------", "\n",
          "\n","input: ")

    print("SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm:")

    x[0] = float(input())
    x[1] = float(input())
    x[2] = float(input())
    x[3] = float(input())

    print("This is: Iris-", names[np.argmax(output(x))], ", with probability: ", np.max(output(x)))

elif option == 2:
    ai_learning()

else:
    print("incorrect input")
