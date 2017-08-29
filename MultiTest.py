from ACJ import ACJ
import numpy as np
import pickle
import random

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    rounds = 16
    length = 100
    errBase = 0.5
    judges = 3
    true = [i+1 for i in range(length)]
    dat1 = true[:]
    dat2 = true[:]
    random.shuffle(dat1)
    random.shuffle(dat2)
    dat = list(zip(dat1, dat2))
    choices = 2
    acj = ACJ(dat, rounds, choices, logPath = "TestLogs", optionNames = ["A", "B"])
    i = 0
    reviewer = "Me"
    with open(r"acj.pkl", "wb") as output_file:
        pickle.dump(acj, output_file)
    del(acj)
    with open(r"acj.pkl", "rb") as input_file:
        acj = pickle.load(input_file)
    while (True):
        print(acj.percentReturned())
        i = i+1
        if (acj.step == 0):
            print(acj.reliability())
        j = acj.nextIDPair();
        if (j == None):
            break
        x = [acj.getScript(l) for l in j]
        res = []
        for i in range(2):
            err = errBase/np.abs(x[0][i]-x[1][i])
            if random.random()<err:
                res.append(x[0][i]<x[1][i])
            else:
                res.append(x[0][i]>x[1][i])
        acj.IDComp(j, result = res, reviewer = reviewer)
        #with open(r"acj.pkl", "wb") as output_file:
        #    pickle.dump(acj, output_file)
        #del(acj)
        #with open(r"acj.pkl", "rb") as input_file:
        #    acj = pickle.load(input_file)
    #acc = np.sum(np.abs(true-rank))/length
    #worst = np.max(np.abs(true-rank))
    print(acj.results())
    print(acj.reliability())
    print(acj.decisionCount("Me"))
    #print(acc)
    #print(worst)
