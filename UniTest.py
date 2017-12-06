from ACJ import ACJ
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    rounds = 15
    maxRounds = 10
    length = 100
    errBase = 0.0
    judges = 2
    true = np.asarray([i+1 for i in range(length)])
    dat = true[:]
    np.random.shuffle(dat)
    acj = ACJ(dat, maxRounds, optionNames = ["Hello"], logPath = "TestLogs")
    reviewers = [i for i in range(10)]

    with open(r"acj.pkl", "wb") as output_file:
        pickle.dump(acj, output_file)
    del(acj)
    with open(r"acj.pkl", "rb") as input_file:
        acj = pickle.load(input_file)
    for i in range(int(rounds*(length/2))):
        reviewer = reviewers[i%10]
        if (acj.step == 0):
            print(acj.reliability())
        j = acj.nextIDPair()
        x = [acj.getScript(k) for k in j]
        #x = acj.nextPair()
        #if (x == None):
        #    break
        err = errBase/np.abs(x[0]-x[1])
        #if (i%10)!=4:
        if True:
            if (random.random() < (np.exp(x[0]-x[1])/(1+np.exp(x[0]-x[1])))):
                res = True
            else:
                res = False
        else:
            res = bool(random.getrandbits(1))
        acj.IDComp(j, result = res, reviewer = reviewer)
        #with open(r"acj.pkl", "wb") as output_file:
        #    pickle.dump(acj, output_file)
        #del(acj)
        #with open(r"acj.pkl", "rb") as input_file:
        #    acj = pickle.load(input_file)
    with open(r"acj.pkl", "wb") as output_file:
        pickle.dump(acj, output_file)
    del(acj)
    with open(r"acj.pkl", "rb") as input_file:
        acj = pickle.load(input_file)

    val = acj.rankings()[1]
    #acc = np.sum(np.abs(true-rank))/length
    #worst = np.max(np.abs(true-rank))
    print(acj.results())
    print(acj.reliability())
    WMS, means, std = acj.WMS()
    for rev in WMS:
        div = abs(WMS[rev]-means)/std
        print(str(rev)+"   "+str(div))
    print(means)
    print(std)
    print(acj.decisionCount(3))
    acj.JSONLog()
    x = []
    y = []
    for t in acj.results()[0]:
        x.append(t[0])
        y.append(t[1])
    print(x)
    plt.plot(x, y, 'ro')
    plt.xlabel('True Value')
    plt.ylabel('Returned Value')
    plt.show()
    #print(acc)
    #print(worst)
