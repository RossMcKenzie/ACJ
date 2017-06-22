import random
import numpy as np
import pickle

class ACJ(object):
    '''Base object to hold comparison data and run algorithm
        script is used to refer to anything that is being ranked with ACJ
        Dat is an array to hold the scripts with rows being [id, script, score, quality]
        Track is an array with each value representing number of times a winner (dim 0) has beaten the loser (dim 1)
    '''
    def __init__(self, data, initVal = 0, swis = 5):
        self.round = 0
        self.update = False
        self.dat = np.zeros((4, len(data)))
        self.dat[0] = np.asarray(range(len(data)))
        self.dat[1] = np.asarray(data)
        self.dat[3, :] = initVal
        self.track = np.zeros((len(data), len(data)))
        self.n = len(data)
        self.swis = swis
        self.roundList = []
        self.step = -1

    def nextRound(self):
        '''Returns next round of pairs'''
        self.round = self.round+1
        if self.round<2:
            self.updateAll()
            print("rand")
            self.roundList = self.randomPairs()
            return self.roundList
        elif self.round<2+self.swis:
            self.updateAll()
            print("swis")
            self.roundList = self.scorePairs()
            return self.roundList
        else:
            print("val")
            #if self.round == 1+swis:
                #self.dat[3] = (1/self.dat[1].size)*self.dat[2][:]
            self.updateAll()
            self.roundList = self.valuePairs()
            return self.roundList
            #return self.scorePairs()

    def nextPair(self):
        '''Returns next pair'''
        self.step = self.step + 1
        if self.step >= len(self.roundList):
            self.nextRound()
            self.step = 0

        return self.roundList[self.step]


    def prob(self, iA):
        '''Retunrs numpy array of the probability of A beating other values
        Based on the Bradley-Terry-Luce model (Bradley and Terry 1952; Luce 1959)'''
        probs = np.exp(self.dat[3][iA]-self.dat[3])/(1+np.exp(self.dat[3][iA]-self.dat[3]))
        return probs

    def updateValue(self, iA):
        '''Updates the value of script A using Newton's Method'''
        scoreA = self.dat[2][iA]
        valA = self.dat[3][iA]
        probA = self.prob(iA)
        x = np.sum(probA)-0.5#Subtract where i = a
        y = np.sum(probA*(1-probA))-0.25#Subtract where i = a
        if x == 0:
            print("FAIl")
            print(probA)
            print(self.dat[3])
            exit()
        #print(self.dat[3])
        return self.dat[3][iA]+((self.dat[2][iA]-x)/y)
        #print(self.dat[3][iA])
        #print("--------")

    def updateAll(self):
        '''Updates the value of all scripts using Newton's Method'''
        newDat = np.zeros(self.dat[3].size)
        for i in self.dat[0]:
            newDat[i] = self.updateValue(i)
        self.dat[3] = newDat[:]

    def randomPairs(self, dat = None):
        '''Returns a list of random pairs from dat'''
        if dat == None:
            dat = self.dat[1]
        shufDat = np.array(dat, copy=True)
        ranPairs = []
        while len(shufDat)>1:
            a = shufDat[0]
            b = shufDat[1]
            shufDat = shufDat[2:]
            ranPairs.append([a,b])
        return ranPairs

    def scorePairs(self, dat = None, scores = None):
        '''Returns random pairs with matching scores or close if no match'''
        if dat == None:
            dat = self.dat
        shuf = np.array(dat[:3], copy=True)
        np.random.shuffle(shuf.T)
        shuf.T
        shuf = shuf[:, np.argsort(shuf[2])]
        pairs = []
        i = 0

        #Pairs matching scores
        while i<(shuf[0].size-1):
            aID = shuf[0][i]
            bID = shuf[0][i+1]
            if (self.track[aID][bID]+self.track[bID][aID])==0 and shuf[2][i]==shuf[2][i+1]:
                pairs.append([shuf[1][i], shuf[1][i+1]])
                shuf = np.delete(shuf, [i, i+1], 1)
            else:
                i = i+1

        #Add on closest score couplings of unmatched scores
        i = 0
        while i<shuf[0].size-1:
            aID = shuf[0][i]
            j = i+1
            while j<shuf[0].size:
                bID = shuf[0][j]
                if (self.track[aID][bID]+self.track[bID][aID])==0:
                    pairs.append([shuf[1][i], shuf[1][j]])
                    shuf = np.delete(shuf, [i, j], 1)
                    break
                else:
                    j = j+1
                if j == shuf[0].size:
                    i = i+1

        return pairs

    def valuePairs(self):
        '''Returns pairs matched by close values'''
        shuf = np.array(self.dat, copy=True)
        np.random.shuffle(shuf.T)
        shuf.T
        pairs = []
        i = 0
        while i<shuf[0].size-1:
            aID = shuf[0][i]
            newShuf = shuf[:, np.argsort(np.abs(shuf[3] - shuf[3][i]))]
            j = 0
            while j<newShuf[0].size:
                bID = newShuf[0][j]
                if (self.track[aID][bID]+self.track[bID][aID])==0 and shuf[1][i]!=newShuf[1][j]:
                    pairs.append([shuf[1][i], newShuf[1][j]])
                    iJ = np.where(shuf[0]==newShuf[0][j])[0][0]
                    shuf = np.delete(shuf, [i, iJ], 1)
                    break
                else:
                    j = j+1
                if j == shuf[0].size:
                    i = i+1

        return pairs

    def rmse(self):
        '''Calculate rmse'''
        pr = np.zeros((self.n, self.n))
        for i in range(self.n):
            pr[i] =  self.dat[3][i]
        prob = np.exp(pr-self.dat[3])/(1+np.exp(pr-self.dat[3]))
        y = 1/np.sqrt(np.sum(prob*(1-prob), axis=1)-0.25)
        return np.sqrt(np.mean(np.square(y)))

    def trueSD(self):
        '''Calculate true standard deviation'''
        sd = np.std(self.dat[3])
        return ((sd**2)/(self.rmse()**2))**(0.5)

    def reliability(self):
        '''Calculates reliability'''
        G = self.trueSD()/self.rmse()
        return (G**2)/(1+(G**2))


    def comp(self, a, b, result = True, update=None):
        '''Adds in a result between a and b where true is a wins and False is b wins'''
        if update == None:
            update = self.update
        iA = np.where(self.dat[1]==a)[0][0]
        iB = np.where(self.dat[1]==b)[0][0]
        if result:
            self.track[iA][iB] = 1
            self.track[iB][iA] = 0
        else:
            self.track[iA][iB] = 0
            self.track[iB][iA] = 1
        self.dat[2][iA] = np.sum(self.track[iA])
        self.dat[2][iB] = np.sum(self.track[iB])

    def rankings(self, value=True):
        '''Returns current rankings
        Default is by value but score can be used'''
        if value:
            return self.dat[:,np.argsort(self.dat[3])]
        else:
            return self.dat[:,np.argsort(self.dat[2])]

if __name__ == "__main__":
    swis = 5
    rounds = 10
    length = 100
    errBase = 0.4
    judges = 3
    true = np.asarray([i+1 for i in range(length)])
    dat = true[:]
    #np.random.shuffle(dat)
    acj = ACJ(dat, swis=swis, initVal=0)

    while (acj.round < rounds):
        x = acj.nextPair();
        err = errBase/np.abs(x[0]-x[1])
        if random.random()<err:
            res = x[0]<x[1]
        else:
            res = x[0]>x[1]
        acj.comp(x[0], x[1], result = res)
        with open(r"acj.pkl", "wb") as output_file:
            pickle.dump(acj, output_file)
        del(acj)
        with open(r"acj.pkl", "rb") as input_file:
            acj = pickle.load(input_file)

    rank = acj.rankings()[1]
    val = acj.rankings()[3]
    acc = np.sum(np.abs(true-rank))/length
    worst = np.max(np.abs(true-rank))
    print(rank)
    print(acj.reliability())
    print(acc)
    print(worst)

'''    for _ in range(rounds):
        print("----------------")
        for x in acj.nextRound():
            #print("%d vs %d" %(x[0], x[1]))
            err = errBase/np.abs(x[0]-x[1])
            if random.random()<err:
                res = x[0]<x[1]
            else:
                res = x[0]>x[1]
            acj.comp(x[0], x[1], result = res)
        print(acj.reliability())

    with open(r"acj.pkl", "wb") as output_file:
        pickle.dump(acj, output_file)

    del(acj)

    with open(r"acj.pkl", "rb") as input_file:
        acj1 = pickle.load(input_file)

    for _ in range(rounds):
        print("----------------")
        for x in acj1.nextRound():
            #print("%d vs %d" %(x[0], x[1]))
            err = errBase/np.abs(x[0]-x[1])
            if random.random()<err:
                res = x[0]<x[1]
            else:
                res = x[0]>x[1]
            acj1.comp(x[0], x[1], result = res)
        print(acj1.reliability())
'''
