import numpy as np

def charToInt(char):
    int = np.zeros(char.shape)
    for i in range(char.shape[0]):
        for j in range(char.shape[1]):
            if char[i,j] == b'draw':
                int[i,j] = -1
            elif char[i,j] == b'zero':
                int[i,j] = 0
            elif char[i,j] == b'one':
                int[i,j] = 1
            elif char[i,j] == b'two':
                int[i,j] = 2
            elif char[i,j] == b'three':
                int[i,j] = 3
            elif char[i,j] == b'four':
                int[i,j] = 4
            elif char[i,j] == b'five':
                int[i,j] = 5
            elif char[i,j] == b'six':
                int[i,j] = 6
            elif char[i,j] == b'seven':
                int[i,j] = 7
            elif char[i,j] == b'eight':
                int[i,j] = 8
            elif char[i,j] == b'nine':
                int[i,j] = 9
            elif char[i,j] == b'ten':
                int[i,j] = 10
            elif char[i,j] == b'eleven':
                int[i,j] = 11
            elif char[i,j] == b'twelve':
                int[i,j] = 12
            elif char[i,j] == b'thirteen':
                int[i,j] = 13
            elif char[i,j] == b'fourteen':
                int[i,j] = 14
            elif char[i,j] == b'fifteen':
                int[i,j] = 15
            elif char[i,j] == b'sixteen':
                int[i,j] = 16
            elif char[i,j] == b'0':
                int[i,j] = 0
            elif char[i,j] == b'1':
                int[i,j] = 1
            elif char[i,j] == b'2':
                int[i,j] = 2
            elif char[i,j] == b'3':
                int[i,j] = 3
            elif char[i,j] == b'4':
                int[i,j] = 4
            elif char[i,j] == b'5':
                int[i,j] = 5
            elif char[i,j] == b'6':
                int[i,j] = 6
            elif char[i,j] == b'7':
                int[i,j] = 7
            elif char[i,j] == b'8':
                int[i,j] = 8
            elif char[i,j] == b'a':
                int[i,j] = 1
            elif char[i,j] == b'b':
                int[i,j] = 2
            elif char[i,j] == b'c':
                int[i,j] = 3
            elif char[i,j] == b'd':
                int[i,j] = 4
            elif char[i,j] == b'e':
                int[i,j] = 5
            elif char[i,j] == b'f':
                int[i,j] = 6
            elif char[i,j] == b'g':
                int[i,j] = 7
            elif char[i,j] == b'h':
                int[i,j] = 8
    
    return int

def splitData(data, split):
    N = data.shape[0]

    idx = np.array(range(N))
    np.random.shuffle(idx)

    test = data[idx[:N//split],:]
    train = data[idx[N//split:],:]

    return train, test

def probabilisticNaiveBayes(train, test):
    print("Probabilistic method")
    classCount = np.zeros(18)
    for i in range(-1, 17):
        classCount[i+1] = np.count_nonzero(train[:,-1][:] == i)
    prior = classCount / train.shape[0]

    likelyhood = np.zeros([6, 8, 18])
    for i in range(likelyhood.shape[0]):
        for j in range(likelyhood.shape[1]):
            for k in range(likelyhood.shape[2]):
                count = 0
                for l in range(train.shape[0]):
                    if train[l,i] == j+1 and train[l,-1] == k-1:
                        count += 1
                
                likelyhood[i,j,k] = count / classCount[k]

    errors = 0
    for d in range(train.shape[0]):
        likelySlice = np.zeros([6, 18])
        for i in range(6):
            likelySlice[i] = likelyhood[i,int(train[d,i]-1),:][:]

        y = np.zeros(18)
        for c in range(18):
            prod = likelySlice[0,c]
            for f in range(1, 6):
                prod *= likelySlice[f,c]

            y[c] = prior[c] * prod
        
        prediction = np.argmax(y) - 1
        if prediction != train[d,-1]:
            errors += 1

        #print("Predicted class:", prediction, "\nActual class:", train[d,-1])

    print("Train error rate:", errors / train.shape[0])

    errors = 0
    for d in range(test.shape[0]):
        likelySlice = np.zeros([6, 18])
        for i in range(6):
            likelySlice[i] = likelyhood[i,int(test[d,i]-1),:][:]

        y = np.zeros(18)
        for c in range(18):
            prod = likelySlice[0,c]
            for f in range(1, 6):
                prod *= likelySlice[f,c]

            y[c] = prior[c] * prod
        
        prediction = np.argmax(y) - 1
        if prediction != test[d,-1]:
            errors += 1

        #print("Predicted class:", prediction, "\nActual class:", test[d,-1])

    print("Test error rate:", errors / test.shape[0])

##############################################################################################################

def basis(x):
    rwk = int(x[0] != x[2]) + int(x[1] != x[3])
    rbk = int(x[2] != x[4]) + int(x[3] != x[5])
    wkr = np.min([abs(x[0] - x[2]), abs(x[1] - x[3])])
    bkr = np.min([abs(x[4] - x[2]), abs(x[5] - x[3])])
    wkbk = np.min([abs(x[0] - x[4]), abs(x[1] - x[5])])

    basis = np.array([[1],
                      [x[0]],
                      [x[1]],
                      [x[2]],
                      [x[3]],
                      [x[4]],
                      [x[5]],
                      [x[0]**2],
                      [x[1]**2],
                      [x[2]**2],
                      [x[3]**2],
                      [x[4]**2],
                      [x[5]**2],
                      [rwk],
                      [rbk],
                      [wkr],
                      [bkr],
                      [wkbk]])
    return basis

def activation(w, x):
    return np.transpose(w) @ basis(x)

def softmax(a, wy, w):
    N = w.shape[0]
    num = np.exp(activation(wy, a))
    
    iter = (np.exp(activation(w[i,:], a)) for i in range(N))
    den = np.sum(np.fromiter(iter, float))

    return num/den

def error(w, data):
    N = data.shape[0]

    iter = (-np.log(softmax(data[i,:-1], w[int(data[i,-1] + 1),:], w)) for i in range(N))
    e = np.sum(np.fromiter(iter, float))

    return e

def gradient(data, w, i):
    N = data.shape[0]
    
    sum = np.sum(np.reshape(basis(data[j,:-1]), 18) * ((data[j,-1] == i - 1) - softmax(data[j,:-1], w[i,:], w)) for j in range(N))

    return sum

def logisticRegression(data, winit, alpha):
    N = data.shape[0]

    w = winit
    wnew = np.zeros([18, 18])
    werror = 0
    wnewerror = 0

    temp, batch = splitData(data, 10)
    werror = error(w, data)
    print("Initial Error:", werror)

    stop = False
    iter = 1
    maxiter = 1000
    try:
        while not stop:
            print("Iteration", iter)

            temp, batch = splitData(data, 10)

            sum = np.array([gradient(batch, w, i) for i in range(18)])
            
            wnew = w + alpha/N * sum
            wnewerror = error(wnew, data)
            print("Error:", wnewerror)

            if abs(werror - wnewerror) < 0.0001 or iter >= maxiter:
                print("Converged")
                stop = True
            elif wnewerror < werror:
                w = wnew
                werror = wnewerror

                alpha *= 1.25
            else:
                alpha *= 0.5

            iter += 1
    except:
        print("Halting early")
        return w, werror
    
    return wnew, wnewerror

def testModel(w, data):
    D = data.shape[0]
    N = w.shape[0]

    errcount = 0
    classErr = np.zeros(18)
    classCount = np.zeros(18)
    for i in range(D):
        y = [softmax(data[i,:-1], w[j,:], w) for j in range(N)]
        classCount[int(data[i,-1] + 1)] += 1

        prediction = np.argmax(y) - 1
        if prediction != data[i,-1]:
            errcount += 1
            classErr[int(data[i,-1] + 1)] += 1
        
    classAcc = np.divide(classErr, classCount)
    print(classAcc)
        
    return errcount / D

def main():
    dataChar = np.genfromtxt('krkopt.data', delimiter=',', dtype=np.character)
    dataInt = charToInt(dataChar)
    
    train, test = splitData(dataInt, 3)

    #probabilisticNaiveBayes(train, test)

    alpha = 1/100

    winit = np.zeros((18, 18))
    
    w, trainerr = logisticRegression(train, winit, alpha)

    print("Error rate by class:")
    trainError = testModel(w, train)
    testError = testModel(w, test)

    print("Linear Regression")
    print("Train error:", trainError)
    print("Test error:", testError)

if __name__ == '__main__':
    main()