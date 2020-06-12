import pickle
import matplotlib.pyplot as plt

prefix = input("Enter the prefix on the file: ")
data_p1 = pickle.load(open("{}data_p1".format(prefix), 'rb'))
data_p2 = pickle.load(open("{}data_p2".format(prefix), 'rb'))

for key in data_p1.keys():
    data1 = data_p1[key]
    data2 = data_p2[key]

    plt.subplot(2, 1, 1)
    plt.plot(range(len(data1)), [i for i in data1])
    plt.ylabel("Total {} p1".format(key))
    plt.subplot(2, 1, 2)
    plt.plot(range(len(data2)), [i for i in data2])
    plt.ylabel("Total {} p2".format(key))
    plt.xlabel("Rounds of training")
    plt.show()