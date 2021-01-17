import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfx=pd.read_csv("xdata.csv")
dfy=pd.read_csv("ydata.csv")

X=dfx.values
Y=dfy.values

X=X[:,1:]
Y=Y[:,1:].reshape((-1,))

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    vals=[]
    m=X.shape[0]

    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))

    vals =sorted(vals)
    vals=vals[:k]

    vals=np.array(vals)

    new_vals=np.unique(vals[:,1],return_counts=True)

    index =new_vals[1].argmax()
    pred=new_vals[0][index]

    return pred

df = pd.read_csv("train.csv")
#Numpy array
data = df.values

X=data[:,1:]
Y=data[:,0]
split=int(0.8*X.shape[0])

X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap="Greys_r")
    plt.show()



pred=knn(X_train,Y_train,X_test[0])


def ask(num):
    num=int(num)
    print("Predicted image :\n")

    drawImg(X_test[num])
    print("Predicted number:\n\n")
    print(Y_test[num])

z=0
print('''
       ScrollWell Bootcamp on Machine Learning using Python
          Day 4 project: Handwritten Digits Recognition
                                        -by Prajwal Ghogare''')

while z!=1:
    x=input("Enter a number...")
    ask(x)
    z=input("\nEnter 1 to stop the Programme||Press Enter to continue")
