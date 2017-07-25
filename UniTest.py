from ACJ import ACJ
import numpy as np
import pickle
import random

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    rounds = 15
    length = 100
    errBase = 0.0
    judges = 3
    true = np.asarray([i+1 for i in range(length)])
    dat = true[:]
    np.random.shuffle(dat)
    acj = ACJ(dat, rounds)
    i = 0
    reviewer = "Me"
    with open(r"acj.pkl", "wb") as output_file:
        pickle.dump(acj, output_file)
    del(acj)
    with open(r"acj.pkl", "rb") as input_file:
        acj = pickle.load(input_file)
    while (True):
        i = i+1
        if (acj.step == 0):
            print(acj.reliability())
        x = acj.nextPair()
        if (x == None):
            break
        err = errBase/np.abs(x[0]-x[1])
        if random.random()<err:
            res = x[0]<x[1]
	    print("hi")
        else:
            res = x[0]>x[1]
        y = []
        y.append(x[1])
        y.append(x[0])
        acj.comp(y, result = not res, reviewer = reviewer)
        #with open(r"acj.pkl", "wb") as output_file:
        #    pickle.dump(acj, output_file)
        #del(acj)
        #with open(r"acj.pkl", "rb") as input_file:
        #    acj = pickle.load(input_file)

    diff = (acj.rankings()[1]-acj.rankings()[1].min())*100/(acj.rankings()[1].max()-acj.rankings()[1].min())
    print(diff)

    rank = acj.rankings()[0]
    val = acj.rankings()[1]
    acc = np.sum(np.abs(true-rank))/length
    worst = np.max(np.abs(true-rank))
    print(rank)
    print(acj.reliability())
    print(acc)
    print(worst)
