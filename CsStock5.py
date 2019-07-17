import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.optim import Optimizer
import math

import os
import matplotlib.pyplot as plt
from sklearn import linear_model
import copy
import time

class Constants1(nn.Module):
    def __init__(self):
        super(Constants1, self).__init__()
        self.linear1 = nn.Linear(1, 10, bias=False)
    def forward(self, x):
        x = self.linear1((x*0.0)+1.0)
        return x

class pastAnalyzer(nn.Module):
    def __init__(self, xSize):
        super(pastAnalyzer, self).__init__()
        ch1, ch2, ch3, ch4 = 10, 5, 5, 20#10
        n1, n2, n3 = 20, 20, 20
        #self.xSize = xSize
        if xSize == 100:
            k1, k2, k3, k4 = 13, 11, 9, 5
            s1, s2, s3, s4 = 1, 2, 2, 1
            k3_2, s3_2 = 7, 2
            #self.conv3_2 = nn.Conv1d(5, ch3, k3_2, stride = s3_2)
        elif xSize == 1000:
            k1, k2, k3, k4 = 13, 11, 9, 7
            s1, s2, s3, s4 = 6, 5, 3, 2
        elif xSize == 25:
            k1, k2, k3, k4 = 11, 9, 5, 3
            s1, s2, s3, s4 = 1, 1, 1, 1
        elif xSize == 5:
            k1, k2, k3, k4 = 3, 3, 1, 1
            s1, s2, s3, s4 = 1, 1, 1, 1
        elif xSize == 1:
            k1, k2, k3, k4 = 1, 1, 1, 1
            s1, s2, s3, s4 = 1, 1, 1, 1
        else:
            print ("Invalid xSize")
            quit()

        latentNum = 10
        self.conv1 = nn.Conv1d(14, ch1, k1, stride = s1, bias=False)
        self.conv2 = nn.Conv1d(ch1, ch2, k2, stride = s2, bias=False) #11
        self.conv3 = nn.Conv1d(ch2, ch3, k3, stride = s3)
        self.conv4 = nn.Conv1d(ch3, ch4, k4, stride = s4)
        self.linearDay = nn.Linear(latentNum, n1)
        #self.linear1 = nn.Linear(ch4, n1)
        #self.linear2 = nn.Linear(n1, n2)
        #self.linear3 = nn.Linear(n2, n3)
        #self.nonlin = torch.tanh
        self.nonlin = F.leaky_relu

    def forward(self, x, dayCode):
        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        x = self.nonlin(x)
        x = self.conv3(x)
        x = self.nonlin(x)
        #if self.xSize == 100:
        #    x = self.conv3_2(x)
        x = self.conv4(x)
        #print (x.shape)
        #quit()
        x = x.view((x.shape[0], 2*10))
        x1 = self.linearDay(dayCode)
        x = x + x1
        x = self.nonlin(x)
        return x

class pastAnalyzer2(nn.Module):
    def __init__(self, xSize):
        super(pastAnalyzer2, self).__init__()
        ch1, ch2, ch3, ch4 = 10, 5, 5, 20#10
        k1, k2, k3, k4 = 13, 11, 9, 7
        n1, n2, n3 = 20, 20, 20
        latentNum = 10
        self.conv1 = nn.Conv1d(14, ch1, k1, stride=6, bias=False)
        self.conv2 = nn.Conv1d(ch1, ch2, k2, stride=5, bias=False) #11
        self.conv3 = nn.Conv1d(ch2, ch3, k3, stride=3)
        self.conv4 = nn.Conv1d(ch3, ch4, k4, stride=2)
        self.linearDay = nn.Linear(latentNum, n1)
        self.linear1 = nn.Linear(ch4, n1)
        self.linear2 = nn.Linear(n1, n2)
        self.linear3 = nn.Linear(n2, n3)
        #self.nonlin = torch.tanh
        self.nonlin = F.leaky_relu

    def forward(self, x, dayCode):
        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        x = self.nonlin(x)
        x = self.conv3(x)
        x = self.nonlin(x)
        x = self.conv4(x)
        x = x.view((x.shape[0], 2*10))
        x1 = self.linearDay(dayCode)
        x = x + x1
        x = self.nonlin(x)
        return x

class catProbNet(nn.Module):
    def __init__(self):
        super(catProbNet, self).__init__()
        self.linear1 = nn.Linear(20, 20)
        self.linear2 = nn.Linear(20, 20)
        self.nonlin = torch.tanh
    def forward(self, x):
        x = self.nonlin(x)
        x = self.linear1(x)
        x = self.nonlin(x)
        x = self.linear2(x)
        x = torch.softmax(x, dim=1)
        return x

class AutoRegressiveLayer(nn.Module):
    def __init__(self, i1, i2, i3, i4, o1, o2, o3, o4, nonlin, doNonlin):
        super(AutoRegressiveLayer, self).__init__()
        self.temp = i1+i2+i3
        self.linear1 = nn.Linear(i1, o1)
        self.linear2 = nn.Linear(i1+i2, o2)
        self.linear3 = nn.Linear(i1+i2+i3, o3)
        self.linear4 = nn.Linear(i1+i2+i3+i4, o4)
        self.nonlin = nonlin
        self.doNonlin = doNonlin
    def forward(self, h1, h2, h3, h4):
        l1 = self.linear1(h1)
        l2 = self.linear2(torch.cat([h1, h2], dim=1))
        l3 = self.linear3(torch.cat([h1, h2, h3], dim=1))
        l4 = self.linear4(torch.cat([h1, h2, h3, h4], dim=1))
        if self.doNonlin:
            l1, l2, l3, l4 = self.nonlin(l1), self.nonlin(l2), self.nonlin(l3), self.nonlin(l4)
        return l1, l2, l3, l4

def calculateProbabilities(y, y1, y2, y3, y4):
    #0: [-3, -2], [-2, 4] 5 units each
    #1: [-6, 5] groups of 5 units each
    #2: [-5, 5] groups of 0.5
    ar1 = torch.arange(0, y.shape[0])
    a1 = (y[:, 3] * 10) + torch.argmax(y[:, 4:], dim=1).float()
    a1 = a1.long()
    a2 = ((y[:, 0] + 2.0) * 5.0)
    #a2 = (a2+0.0001) * 0.9999
    #a2 = torch.floor(a2).long()
    a2 = torch.tensor(np.floor(a2.cpu().data.numpy())).long()
    a2[a2<-2] = 30
    a3 = ((y[:, 2] + 5.0) * 2.0)
    #a3 = (a4+0.0001) * 0.9999
    #a3 = torch.floor(a3).long()
    a3 = torch.tensor(np.floor(a3.cpu().data.numpy())).long()
    a4 = ((y[:, 1] + 6.0) * 5.0)
    #a4 = (a4+0.0001) * 0.9999
    #a4 = torch.floor(a4).long()
    a4 = torch.tensor(np.floor(a2.cpu().data.numpy())).long()


    p1 = y1[ar1, a1]
    p2 = y2[ar1, a2]
    p3 = y3[ar1, a3]
    p4 = y4[ar1, a4]

    '''
    for a in np.arange(1000):
        yVar = y3[a].data.numpy()
        xVar = np.arange(yVar.shape[0])
        plt.scatter(xVar, yVar)
    #plt.yscale('log')
    plt.show()
    quit()
    '''
    #print (y4)
    #quit()
    return p1, p2, p3, p4

def softNonLin(x):
    return torch.softmax(x, dim=1)

class AutoRegressiveModel(nn.Module):
    def __init__(self, numLayer, xSize):
        super(AutoRegressiveModel, self).__init__()
        numLayer = 2
        self.analyzer = pastAnalyzer(xSize)
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(1, 10)
        self.nonlin = torch.tanh
        self.finalnonlin = softNonLin
        self.numLayer = numLayer
        self.layer1 = AutoRegressiveLayer(20, 11, 10, 10, 10, 10, 10, 10, self.nonlin, True)
        if self.numLayer > 2:
            self.layer2 = AutoRegressiveLayer(10, 10, 10, 10, 10, 10, 10, 10, self.nonlin, True)
        if self.numLayer > 3:
            self.layer3 = AutoRegressiveLayer(10, 10, 10, 10, 10, 10, 10, 10, self.nonlin, True)
        self.layer4 = AutoRegressiveLayer(10, 10, 10, 10, 20, 31, 20, 55, self.finalnonlin, True)

        self.fake1 = nn.Linear(1, 20)

    def forward(self, x, y, code):
        #codePartNum = 1
        #yNum = 1
        #sampleNum = 21
        #code[:, codePartNum] = code[:, codePartNum] - 1.0
        #for z in range(0, 3):
            #code[:, codePartNum] = code[:, codePartNum] + 1.0

        newCode = self.analyzer(x, code)


        #newCode = self.fake1(torch.zeros((x.shape[0], 1)))

        #print (newCode[0])
        #print (newCode[100])
        #print (newCode[200])
        #quit()
        #for a in range(0, 20):
        #    newCode[:, a] = torch.mean(newCode[:, a])


        y1 = y[:, 3:]
        y2 = self.linear1(y[:, torch.tensor([0])])
        y3 = self.linear1(y[:, torch.tensor([2])])
        y2 = self.nonlin(y2)
        y3 = self.nonlin(y3)

        y1, y2, y3, y4 = self.layer1(newCode, y1, y2, y3)
        if self.numLayer > 2:
            y1, y2, y3, y4 = self.layer2(y1, y2, y3, y4)
        if self.numLayer > 3:
            y1, y2, y3, y4 = self.layer3(y1, y2, y3, y4)
        y1, y2, y3, y4 = self.layer4(y1, y2, y3, y4)

        #for a in range(0, 10):
        #    plt.plot(np.log(y4[a*10].data.numpy()))
        #plt.show()
        #quit()
        #print (torch.max(y1, dim=0)[0]/torch.min(y1, dim=0)[0])
        #print (torch.min(y1, dim=0)[0])
        #quit()
        #print (time.time()-1562013000.0)
        p1, p2, p3, p4 = calculateProbabilities(y, y1, y2, y3, y4)
        #allY = [y1, y2, y3, y4]
        #plt.plot(allY[yNum][sampleNum].data.numpy())

        #plt.yscale('log')
        #plt.show()
        #print (y2[0])
        #quit()
        #print (time.time()-1562013000.0)
        p = torch.log(p1) + torch.log(p2) + torch.log(p3) + torch.log(p4)
        loss = -torch.mean(p)
        #print ()
        #print (loss)
        #if loss > 8.0:
        #    pH = p.data.numpy()
        #    plt.hist(pH)
        #    plt.show()
        #    quit()
            #print (loss)
        #quit()
        return loss

def saveStartPoints(stockID):
    data = np.load('./InputData/' + stockID + '_NumericMessages.npy')
    len1 = data.shape[0] - 1100
    ar = np.arange(len1)
    startPoints = []
    for a in range(0, 100000):
        if a % 1000 == 0:
            print (a)
        startPoint = np.random.choice(ar, size=1)
        startPoints.append(startPoint)
    np.save("InputData/" + stockID + "_startPoints.npy", startPoints)

def packData(data, packSize):
    packs = []
    for a in range(0, data.shape[0]-100):
        if a % 1000 == 0:
            print (a//1000)
            print (data.shape[0])
        packs.append(data[a:a+100])
    np.save("GANData/packs.npy", packs)

def makeBatch(data, startPoint, batchSize, xSize):
    ar1 = np.arange(xSize + 1)
    ar2 = np.tile(ar1, batchSize)
    ar3 = np.arange(batchSize)
    ar4 = np.repeat(ar3, xSize + 1)
    ar5 = ar2 + ar4 + startPoint

    x = np.copy(data[ar5])
    x = np.reshape(x, (batchSize, xSize + 1, data.shape[1]))
    #x = []
    #for a in range(0, batchSize):
    #    x.append(np.copy(data[a+startPoint:a+startPoint+101]))
    x = torch.tensor(x).float()
    x = x.permute((0, 2, 1))

    y = x[:, :, -1]
    x = x[:, :, :-1]
    fake = torch.zeros((batchSize, 1))
    return x, y, fake

def makeBatch2(data, startPoint, batchSize):
    x = []
    for a in range(0, batchSize):
        x.append(np.copy(data[(a*10)+startPoint:(a*10)+startPoint+1001]))
    x = torch.tensor(x).float()
    x = x.permute((0, 2, 1))
    return x

def preProcessing(stockID):
    data = np.load('./InputData/' + stockID + '_NumericMessages.npy')
    d1 = data[:, 1]
    if stockID == "SPY":
        d1[d1>500] = 500
        data[:, 1] = d1
    #d1[d1>250] = 250
    #d1[d1<50] = 50

    data[:, 1] = d1
    data[:, 2] = np.log(1+data[:, 2])
    data[1:, 0] = data[1:, 0] - data[:-1, 0]
    data[1:, 0] = np.log(data[1:, 0]+1)
    data = data[1:]
    data[:, 3][data[:, 3] != 1] = 0
    for a in range(0, 3):
        data[:, a] = data[:, a] - np.mean(data[:, a])
        if a == 1:
            data[:, a] = data[:, a] * ((np.log(np.abs(data[:, a]) + 0.1) - np.log(np.abs(0.1))) / np.abs(data[:, a]))
            #data[:, a] = data[:, a] - np.mean(data[:, a])
            #data[:, a] = data[:, a] * ((np.log(np.abs(data[:, a]) + 1.0) - np.log(np.abs(1.0))) / np.abs(data[:, a]))
        data[:, a] = data[:, a] / (np.mean(data[:, a]**2.0)**0.5)

    plt.hist(data[:, 1])
    plt.yscale('log')
    plt.show()

    np.save('./InputData/' + stockID + '_NormalizedNumericMessages.npy', data)

def trainARM():
    data = np.load('./NormalizedNumericMessages.npy')
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    #model = torch.load('./Models/ARM/ARM_2.pt')
    model = AutoRegressiveModel()
    con = Constants1()
    learningRate = 0.01
    optimizer1 = torch.optim.RMSprop(model.parameters(), lr = learningRate)
    #learningRate = 0.01
    #optimizer2 = torch.optim.RMSprop(con.parameters(), lr = learningRate)
    batchSize = 10000
    iters = 10000
    lossTotal = 0.0
    startPoints = np.load("GANData/startPoints.npy")
    losses = []
    for a in range(0, iters):

        startPoint = int(startPoints[a][0]) #TODO USE THIS!
        #startPoint = [1000 * a]
        #if startPoint[0] > (data.shape[0] - 1200):
        #    startPoint[0] = 1000
        #batch = torch.tensor(startPoint + np.array(batchSize//2))

        try:
            x1 = makeBatch2(data, startPoint, batchSize)
            y1 = x1[:, :, -1]
            x1 = x1[:, :, :-1]

            fake = torch.zeros((batchSize, 1))
            code = con(fake) * 0.0
            #print (testModel(x1, code)[0])
            #quit()

            loss = model(x1, y1, code)
            #lossTotal += loss
            losses.append(loss.data.numpy())
            #print (a)
            print (loss)
            #if a == 1:
            #   quit()

            optimizer1.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
            optimizer1.step()

            #quit()
            if (a+1)%100==0:
                print (a)
                print (np.mean(losses[-100:]))
                #np.save("./Results/losses.npy", losses)
                torch.save(model, './Models/ARM/ARM_5.pt')
        except:
            print ("Fail")
    torch.save(model, './Models/ARM/ARM_5.pt')
    #plt.plot(losses)
    #plt.show()
    #print (lossTotal)

def movingAverage(ar, num, type):
    data = []
    for a in range(0, num):
        data.append(ar[a:-num+a])
    if type == "mean":
        ar2 = np.mean(np.array(data), axis=0)
    if (type == "median") or (type == "med"):
        ar2 = np.median(np.array(data), axis=0)
    return ar2

def saveTimes():
    data1 = np.load('./InputData/' + 'SPY' + '_NumericMessages.npy')
    data2 = np.load('./InputData/NumericMessages.npy')
    times1 = data1[500::1000, 0].astype(float)
    times2 = data2[500::1000, 0].astype(float)
    times = [times1, times2]
    np.save('./InputData/Times1.npy', times)


def XYCodes(mode='Multi'):
    if (mode == "Multi") or ((mode == "multi") or (mode == "multiple")):
        scores = []
        for codePredict in [0, 1]:
            for codeNum in range(0, 5):
                numBack = 5
                #codePredict = 0
                #codeNum = 0
                args = np.load('./InputResults/TimeArgs.npy')
                codes1 = np.load('./InputResults/ARM13_1codes.npy')[args[0], :5]
                codes2 = np.load('./InputResults/ARM13_2codes.npy')[args[1], :5]

                #codes1[:, 1:] = 0
                ar = list(np.arange(5))
                ar.remove(codeNum)
                ar = np.array(ar)
                if codePredict == 0:
                    #XCode = np.concatenate([codes1[:, ar], codes2], axis=1)
                    YCode = np.copy(codes1[:, codeNum])
                    #codes1[:, ar] = 0
                    #codes1[:] = 0
                    #codes2[:] = 0
                else:
                    #XCode = np.concatenate([codes1, codes2[:, ar]], axis=1)
                    YCode = np.copy(codes2[:, codeNum])
                    #codes2[:, ar] = 0
                    #codes1[:] = 0
                    #codes2[:] = 0
                XCode = np.concatenate([codes1, codes2], axis=1)
                X = []
                Y = []
                for a in range(0, codes1.shape[0]-numBack):
                    X.append(XCode[a:a+numBack, :].flatten())
                    Y.append(YCode[a+numBack])

                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(X, Y)
                score = reg.score(X, Y)
                scores.append(score)
                #print (np.reshape(reg.coef_, (numBack, 10)).T)
                #quit()
                Y1 = reg.predict(X)
                plt.plot(Y[200:250])
                plt.plot(Y1[200:250])
                plt.show()
        print (np.mean(scores))
        print (scores)
    elif (mode == "Single") or (mode == "single"):
        scores = []
        for codeNum in range(0, 5):
            numBack = 5
            #codePredict = 0
            #codeNum = 0
            #args = np.load('./InputResults/TimeArgs.npy')
            codes = np.load('./InputResults/ARM1codes.npy')
            #codes1[:, 1:] = 0
            ar = list(np.arange(5))
            ar.remove(codeNum)
            ar = np.array(ar)
            YCode = np.copy(codes[:, codeNum])
            XCode = codes
            X = []
            Y = []
            for a in range(0, codes.shape[0]-numBack):
                X.append(XCode[a:a+numBack, :].flatten())
                Y.append(YCode[a+numBack])

            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X, Y)
            score = reg.score(X, Y)
            scores.append(score)
            #print (np.reshape(reg.coef_, (numBack, 10)).T)
            #quit()
            Y1 = reg.predict(X)
            plt.plot(Y[200:250])
            plt.plot(Y1[200:250])
            plt.show()
        print (np.mean(scores))
        print (scores)
    else:
        print ("Error")
        quit()



def doTimeArgs():
    times = np.load('./InputData/Times1.npy')
    #losses1 = np.load("./Results/ARM12losses.npy")
    #losses2 = np.load("./Results/ARM9losses.npy")
    losses1 = np.load('./InputResults/ARM13_1codes.npy')[:, 0]
    losses2 = np.load('./InputResults/ARM13_2codes.npy')[:, 0]
    newTimes = []
    newCodes1 = []
    newCodes2 = []
    newLosses1 = []
    newLosses2 = []
    sizes = [losses1.shape[0], losses2.shape[0]]
    args = [[], []]
    a = [0, 0]
    done = False
    while not done:
        #print (a)
        #if (a[0]%1000) == 0:
        #    print (a[0]//1000)
        next1 = times[0][a[0]+1]
        next2 = times[1][a[1]+1]
        if next1 > next2:
            larger = 0
            time = next1

        else:
            larger = 1
            time = next2
        smaller = (larger+1)%2
        a[larger] += 1
        if (a[0] < sizes[0]) or (a[1] < sizes[1]):
            isLarger = False
            while not isLarger:
                if a[smaller] + 1 < sizes[smaller]:
                    if times[smaller][a[smaller]+1] < time:
                        a[smaller] += 1
                    else:
                        isLarger = True
                else:
                    a[smaller] = a[smaller] + 1
                    done = True
                    isLarger = True
            if (a[0] < sizes[0]) and (a[1] < sizes[1]):
                newTimes.append(time)
                args[0].append(a[0])
                args[1].append(a[1])
                #newLosses1.append(losses1[a[0]])
                #newLosses2.append(losses2[a[1]])
            else:
                done = True
        else:
            done = True

    np.save('./InputResults/TimeArgs.npy', args)
    #print (newTimes[:10])
    #quit()

    #plt.plot(movingAverage(newLosses1, 100, "mean"))#[500:1000])
    #plt.plot(movingAverage(newLosses2, 100, "mean"))#[500:1000])
    #plt.plot(newLosses1[600:700])
    #plt.plot(newLosses2[600:700])
    #plt.scatter(newLosses1, newLosses2)
    #plt.show()
    quit()


    losses1 = losses1 - np.mean(losses1)
    losses2 = losses2 - np.mean(losses2)

    ar1 = np.arange(losses1.shape[0])
    ar2 = np.arange(losses2.shape[0])
    ar1 = ar1 / (np.mean(ar1**2.0)**0.5)
    ar2 = ar2 / (np.mean(ar2**2.0)**0.5)
    losses1 = losses1 - (np.mean(ar1*losses1) * ar1)
    losses2 = losses2 - (np.mean(ar2*losses2) * ar2)

    x1 = times[0][np.arange(losses1.shape[0])]
    x2 = times[1][np.arange(losses2.shape[0])]
    codes1 = np.array([x1, losses1]).T
    codes2 = np.array([x2, losses2]).T
    s = 100
    #codes1.shape[1]
    codes3 = np.zeros((codes1.shape[0]-s, codes1.shape[1]))
    codes4 = np.zeros((codes2.shape[0]-s, codes2.shape[1]))
    for a in range(0, s):
        codes3 = codes3 + codes1[a:a-s, :]
        codes4 = codes4 + codes2[a:a-s, :]
    codes3 = codes3 / s
    codes4 = codes4 / s
    plt.plot(codes3.T[0], codes3.T[1])
    plt.plot(codes4.T[0], codes4.T[1])
    plt.show()


def codeBatchPredictor():
    #losses1 = np.load("./Results/ARMBatch10_2_losses.npy")
    #losses2 = np.load("./Results/ARMBatch10_losses.npy")
    #plt.plot(losses1[:10000])
    #plt.plot(losses2[:10000])
    #plt.show()
    #quit()

    codes_1 = np.load("./Results/M2_10B2_TestCodes.npy")
    codesPredict = []
    #plt.plot(codes_1[:500, 1])
    #plt.show()
    #quit()
    #codes_1 = np.load('./InputResults/ARMBatch10_codes.npy')
    #codes_2 = np.load("./InputResults/ARMBatch100_codes.npy")
    for codeNum in range(0, 5):
        codes1 = codes_1[:, codeNum]
        #codes2 = codes_2[start:end, codeNum]
        codes1 = codes1 - np.mean(codes1)
        #codes2 = codes2 - np.mean(codes2)
        x = []
        y = []
        for a in range(0, codes1.shape[0]-10):
            x.append(codes1[a:a+10])
            y.append([codes1[a+10]])

        from sklearn.linear_model import LinearRegression
        x, y = np.array(x), np.array(y)
        #x1, x2 = x1*0.0, x2*0.0
        #reg1 = LinearRegression().fit(x1, y)
        reg = LinearRegression().fit(x, y)
        y2 = reg.predict(x)
        #print ('a')
        #print (reg1.score(x1, y))
        print (reg.score(x, y))
        #print (reg1.coef_)
        #print (reg.coef_)
        print (np.sum(np.abs(reg.coef_[0][:-1])))
        codesPredict.append(y2)
    codesPredict = np.array(codesPredict).T
    np.save("InputResults/CodesPred2_M2.npy", codesPredict)
    #print (reg2.coef_[0][:-1])
        #plt.plot(reg2.coef_[0][:-1])
    #plt.show()
    #quit()
    #plt.plot(np.arange(codes1.shape[0])[15000:], codes1[15000:, codeNum])
    #plt.plot(np.arange(end-start)[1500:]*10, codes2[1500:, codeNum])
    #plt.show()
    '''
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, Y)
    score = reg.score(X, Y)
    scores.append(score)
    #print (np.reshape(reg.coef_, (numBack, 10)).T)
    #quit()
    Y1 = reg.predict(X)
    plt.plot(Y[200:250])
    plt.plot(Y1[200:250])
    plt.show()
    '''

def doPlots():
    data = np.load('./InputData/NumericMessages.npy')
    times = data[:, 0]
    diffs = times[1:] - times[:-1]
    diffs[diffs==0.0] = np.mean(diffs)
    print (np.mean(diffs))
    diffs = 1.0/diffs
    print (1.0/np.mean(diffs))
    #print (data[0, 0])
    #print (data[-1, 0])
    quit()


    losses = np.load("./Results/VA1_B100_L5_X100_losses.npy")
    losses = movingAverage(losses, 1000, "med")[60000:]
    plt.plot(losses)
    plt.show()
    quit()

    losses1 = np.load("./Results/ARMBatch10_3P_losses.npy")
    losses2 = np.load("./Results/ARMBatch10_4P_losses.npy")
    print (np.mean(losses1))
    print (np.mean(losses2))
    plt.plot(losses1)
    plt.plot(losses2)
    plt.show()
    quit()


    #'''
    codes1 = np.load('./InputResults/ARMBatch10_codes.npy')
    codes2 = np.load("./InputResults/ARMBatch100_codes.npy")
    start = 25000
    codeNum = 2
    end = start + (codes1.shape[0] // 10)
    plt.plot(np.arange(codes1.shape[0])[15000:], codes1[15000:, codeNum])
    plt.plot(np.arange(end-start)[1500:]*10, codes2[start:end][1500:, codeNum])
    #print (codes1.shape)
    #plt.plot(losses[15700:])
    #plt.plot(losses[16075:16110])
    #plt.plot(codes[16075:16110, 0]*10.0)
    plt.show()
    quit()
    #'''

    losses1 = np.load("./Results/ARMBatch10_losses.npy")
    losses2 = np.load("./Results/ARMBatch100_losses.npy")
    s = 10
    losses3 = np.zeros(losses1.shape[0] - s)
    for a in range(0, s):
        losses3 = losses3 + losses1[a:a-s]
    losses3 = losses3 / s
    start = 25000
    end = start + (losses3.shape[0] // 10)
    #print (np.mean(losses3[15000:]))
    #print (np.mean(losses2[start:end][1500:]))
    plt.plot(np.arange(losses3.shape[0])[15000:], losses3[15000:])
    plt.plot(np.arange(end-start)[1500:]*10, losses2[start:end][1500:])
    #print (codes1.shape)
    #plt.plot(losses[15700:])
    #plt.plot(losses[16075:16110])
    #plt.plot(codes[16075:16110, 0]*10.0)
    plt.show()
    quit()



    losses = np.load("./Results/ARMBatch100_losses.npy")
    codes = np.load('./InputResults/ARMBatch100_codes.npy')
    #plt.plot(losses[25700:25720])
    #plt.plot(codes[25700:25720])
    plt.plot(losses[5000:20000])
    #plt.plot(codes)
    #plt.plot(losses[15600:15800])
    #plt.plot(codes[15600:15800])
    #plt.plot(losses[15425:15450])
    #plt.plot(codes[15425:15450])
    #plt.plot(losses[10500:10530])
    #plt.plot(codes[10500:10530])
    plt.show()
    quit()
    codes = np.load('./InputResults/ARM13_2codes.npy')
    losses = np.load('./Results/ARM13_2losses.npy')
    #plt.plot(losses[2665:2575])
    #plt.plot(codes[2565:2575])
    #plt.plot(losses[2665:2575])
    #plt.plot(codes[2565:257])

    #plt.plot(losses*0.0)
    #plt.plot()
    plt.show()
    quit()


    '''
    #[0, 3, 4]
    losses1 = np.load("./Results/ARM13_1losses.npy")
    codes1 = np.load('./InputResults/ARM13_1codes.npy')
    losses2 = np.load("./Results/ARM13_2losses.npy")
    codes2 = np.load('./InputResults/ARM13_2codes.npy')
    #plt.plot(codes1)
    #plt.plot(codes2)
    #plt.show()

    #plt.plot(codes1[:, 2])
    #plt.plot(codes2[:, 2])
    plt.show()
    #print (np.mean(codes1, axis=0)[:5])
    #print (np.mean(codes2, axis=0)[:5])
    quit()
    #'''


    times = np.load('./InputData/Times1.npy')
    #losses1 = np.load("./Results/ARM12losses.npy")#[-100:]
    #losses2 = np.load("./Results/ARM9losses.npy")#[-100:]
    losses1 = np.load('./InputResults/ARM13_1codes.npy')[:, 0][90:-30]
    losses2 = np.load('./InputResults/ARM13_2codes.npy')[:, 0][90:-30]

    losses1 = losses1 - np.mean(losses1)
    losses2 = losses2 - np.mean(losses2)

    ar1 = np.arange(losses1.shape[0])
    ar2 = np.arange(losses2.shape[0])
    ar1 = ar1 / (np.mean(ar1**2.0)**0.5)
    ar2 = ar2 / (np.mean(ar2**2.0)**0.5)
    losses1 = losses1 - (np.mean(ar1*losses1) * ar1)
    losses2 = losses2 - (np.mean(ar2*losses2) * ar2)

    x1 = times[0][np.arange(losses1.shape[0])]
    x2 = times[1][np.arange(losses2.shape[0])]
    codes1 = np.array([x1, losses1]).T
    codes2 = np.array([x2, losses2]).T
    s = 100
    #codes1.shape[1]
    codes3 = np.zeros((codes1.shape[0]-s, codes1.shape[1]))
    codes4 = np.zeros((codes2.shape[0]-s, codes2.shape[1]))
    for a in range(0, s):
        codes3 = codes3 + codes1[a:a-s, :]
        codes4 = codes4 + codes2[a:a-s, :]
    codes3 = codes3 / s
    codes4 = codes4 / s
    plt.plot(codes3.T[0], codes3.T[1])
    plt.plot(codes4.T[0], codes4.T[1])
    plt.show()


    #x1 = np.arange(losses1.shape[0])
    #x2 = np.arange(losses2.shape[0]) * (losses1.shape[0]/losses2.shape[0])
    #plt.plot(x1[90:-30], losses1[90:-30])
    #plt.plot(x2[90:-30], losses2[90:-30])
    #plt.plot(x1[90:-30], losses1[90:-30])
    #plt.plot(x2[90:-30], losses2[90:-30])
    #plt.show()
    quit()
    '''
    losses = np.load("./Results/ARM5losses.npy")[2657:] #Full
    plt.plot(losses)
    plt.show()
    quit()
    '''
    '''
    codes = np.load('./InputResults/ARM10codes.npy')[:, :2]
    s = 100
    codes2 = np.zeros((codes.shape[0]-s, codes.shape[1]))
    for a in range(0, s):
        codes2 = codes2 + codes[a:a-s, :]
    codes2 = codes2 / s

    plt.plot(codes2)
    plt.show()
    quit()
    '''


    losses = np.load("./Results/ARM5losses.npy")#[2500:] #Full
    losses3 = np.load("./Results/ARM6losses.npy")#[2500:] #No Cheat
    losses1 = np.load("./Results/ARM1losses.npy")#[2500:] #No Latent
    losses9 = np.load("./Results/ARM9losses.npy")#[2500:] #Only 5 latent
    losses10 = np.load("./Results/ARM10losses.npy")#[2500:] #Only 2 latent
    losses5_10 = np.load("./Results/ARM5losses_10UpdateLatent.npy")
    print (np.mean(losses10))
    print (np.mean(losses9))
    print (np.mean(losses))
    print (np.mean(losses5_10))
    #plt.plot(losses10-losses9)
    #plt.plot(losses9-losses)
    plt.plot(losses-losses5_10)
    #plt.plot(losses*0.0)
    #plt.plot()
    plt.show()
    quit()


    codes = np.load('./InputResults/ARM1codes.npy')

    s = 100
    codes2 = np.zeros((codes.shape[0]-s, codes.shape[1]))
    for a in range(0, s):
        codes2 = codes2 + codes[a:a-s, :]
    codes2 = codes2 / s



    plt.plot(codes2)
    plt.show()
    quit()
    #print (codes.shape)
    for a in range(0, 10):
        codes[:, a] = codes[:, a] - np.mean(codes[:, a])
        codes[:, a] = codes[:, a] / (np.sum(codes[:, a]**2.0)**0.5)
    matrix = np.matmul(codes.T, codes)
    print (matrix[0])
    #plt.imshow(matrix)
    #plt.show()
    #print (np.sum(np.abs(matrix)))
    #print (matrix)
    quit()
    '''
    losses = np.load("./Results/ARM5losses.npy")#[2500:]
    losses3 = np.load("./Results/ARM6losses.npy")#[2500:]
    #losses1 = np.load("./Results/ARM1losses.npy")[2500:]
    print (np.mean(losses))
    print (np.mean(losses3))
    #print (np.mean(losses1))
    #print (losses1[:5])
    #print (losses3)
    #quit()
    #plt.plot(losses3)
    #plt.plot(losses)
    adv = losses3-losses
    adv = np.sign(adv) * np.abs(adv)
    plt.plot(losses3)
    plt.plot(adv)
    plt.show()
    quit()
    '''


    '''
    losses3 = np.load("./Results/ARM3losses.npy")
    losses1 = np.load("./Results/ARM1losses.npy")
    losses3 = losses3 - 4
    #plt.plot(losses1)
    plt.hist(losses3, bins=100)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    quit()
    #'''

    '''
    codes = np.load('./InputResults/ARM1codes.npy')#[250:2250, :]
    s = 100
    codes2 = np.zeros((codes.shape[0]-s, codes.shape[1]))
    for a in range(0, s):
        codes2 = codes2 + codes[a:a-s, :]
    codes2 = codes2 / s
    codes2 = codes[s:, :] - codes[:-s, :]
    codes3 = np.zeros((codes2.shape[0]-s, codes2.shape[1]))
    for a in range(0, s):
        codes3 = codes3 + codes2[a:a-s, :]
    codes3 = codes3 / s
    plt.plot(codes3)
    plt.show()
    quit()
    '''


    '''
    batchSize = 1000
    ar1 = np.arange(101)
    ar2 = np.tile(ar1, batchSize)
    ar3 = np.arange(batchSize)
    ar4 = np.repeat(ar3, 101)
    ar5 = ar2 + ar4
    data = [np.arange(2000), np.zeros(2000)]
    data = np.array(data).T
    data = data[ar5]
    data = np.reshape(data, (1000, 101, data.shape[1]))
    '''

    '''
    #losses1 = np.load("./Results/losses.npy")
    #losses2 = np.load("./Results/lossesARM9.npy")
    #losses3 = np.load("./Results/lossesARM9_OldCode2.npy")
    losses1 = np.load("./Results/losses2ARM_6.npy")#[150:]
    losses2 = np.load("./Results/losses2ARM_7.npy")#[150:]
    losses3 = np.load("./Results/losses2ARM_8.npy")#[150:]
    len = min(losses1.shape[0], losses2.shape[0], losses3.shape[0])
    #len = 200
    #adv1 = losses1 - losses2
    adv2 = losses1[:700] - losses3[:700]
    losses1 = losses1 - np.mean(losses1)
    losses1 = losses1 * np.mean(np.abs(adv2)) / np.mean(np.abs(losses1))
    plt.plot(losses1[:200])#[:len])
    #plt.plot(adv1)#[:len])
    plt.plot(adv2[:200])#[:len])
    plt.show()
    quit()
    #'''

    '''
    #losses2 = np.load("./Results/ARM1losses.npy")[250:2250] #Latents
    losses2 = np.load("./Results/ARM1losses2.npy")[250:2250] #Latents but oldCode
    losses3 = np.load("./Results/ARM3losses.npy")[250:2250] #No Latents
    losses2 = losses3 - losses2
    losses3 = losses3 - np.mean(losses3)
    plt.plot(losses3)
    plt.plot(losses2)
    plt.show()
    quit()
    codes = np.load('./InputResults/ARM1codes.npy')#[250:2250, :]
    codes1 = codes[:, 0]

    losses2 = losses3 - losses2

    losses2 = movingAverage(losses2, 10, "mean")
    losses3 = movingAverage(losses3, 10, "mean")
    codes1 = movingAverage(codes1, 10, "mean")

    losses3 = losses3 - np.mean(losses3) #+ 1
    #losses2 = losses2 / np.max(np.abs(losses2))
    #losses3 = losses3 / np.max(np.abs(losses3))


    #losses3 = np.abs(losses3[100:] - losses3[:-100])
    #losses2 = losses2[50:-50]


    plt.plot(losses3)
    plt.plot(losses2) #advantage
    #plt.plot(codes1)
    plt.plot(np.zeros(losses2.shape[0]))
    #plt.scatter(losses3, losses2)
    plt.show()
    quit()
    '''

    '''
    #codes = np.load("./Results/codesARM9.npy")
    #codes = np.load("./Results/codes2ARM7.npy")
    codes = np.load('./InputResults/codes2.npy')
    #codes = np.abs(codes[1:] - codes[:-1])
    code1 = codes[:, 0]
    code2 = codes[:, 1]

    code1 = code1 - np.mean(code1)
    code1 = code1 / (np.sum(code1**2.0) ** 0.5)
    code2 = code2 - np.mean(code2)
    code2 = code2 / (np.sum(code2**2.0) ** 0.5)

    #plt.plot(code1)
    #plt.plot(code2)

    import scipy
    import scipy.stats#.spearmanr as spearman

    print (scipy.stats.spearmanr(code1[:-1], code1[1:]))


    #print (np.sum(code1*code2))
    #plt.scatter(code1, code2)
    #plt.show()
    quit()
    '''

    '''
    #plt.plot(np.abs(codes[85:130]))
    #sizes = np.sum(codes**2.0, axis=0)
    #print (sizes[np.array([0, 5, 6, 7])])
    #quit()
    #codes = codes[:-5] + codes[1:-4] + codes[2:-3] + codes[3:-2] + codes[4:-1] + codes[5:]

    #plt.plot(codes[150:, np.array([0, 5, 6, 7])])
    plt.show()
    quit()
    #'''


def trainLatentARM(updateModel=False, latentIters=5, randomStartPoints=True, doCuda=True, name=""):
    batchSize = 100
    #data = np.load('./InputData/NormalizedNumericMessages.npy')
    #data1 = np.load('./InputData/SPY_NormalizedNumericMessages.npy')
    data2 = np.load('./InputData/NormalizedNumericMessages.npy')
    #print (data2.shape)
    #quit()
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    xSize = 25
    model = AutoRegressiveModel(4, xSize)
    #modelName = './Models/ARM4/' + name + ".pt"
    #model = torch.load(modelName)
    #model = torch.load( './Models/ARM3/ARM9.pt')
    #updateModel = False
    oldCodeTest = False
    con = Constants1()
    if doCuda:
        model.cuda()
        con = con.cuda()
    learningRate = 0.001
    optimizer1 = torch.optim.RMSprop(model.parameters(), lr = learningRate)
    #learningRate = 1000.0
    #optimizer1 = torch.optim.SGD(model.parameters(), lr = learningRate)
    learningRate = 0.05
    optimizer2 = torch.optim.RMSprop(con.parameters(), lr = learningRate)
    #codes2 = torch.tensor(np.load("InputResults/CodesPred_4.npy")).float()
    iters =  100000
    #iters = 10000000
    #latentIters = 10
    lossTotal = 0.0
    len1 = 0
    #startPoints = np.load("./InputData/startPoints.npy")
    #startPoints1 = np.load("./InputData/SPY_startPoints.npy")
    startPoints2 = np.load("./InputData/startPoints.npy")
    #previusCodes = np.load('./InputResults/codes.npy')
    codes2 = np.load("./InputResults/CodesPred2_M2.npy")
    losses = []
    codes = []
    for a in range(0, iters):
        #print (a)
        if randomStartPoints:
            #if a % 2 == 0:
            #startPoint = int(startPoints1[a][0])
            #else:
            startPoint = int(startPoints2[a][0])
        else:
            startPoint = batchSize * (a+170000+10) #(a + 250000 + 10)
        #print (a)
        skipIter = False
        try:
            #if a % 2 == 0:
            #x1, y1, fake = makeBatch(data1, startPoint, batchSize, xSize)
            #else:
            x1, y1, fake = makeBatch(data2, startPoint, batchSize, xSize)
            if doCuda:
                x1, y1, fake = x1.cuda(), y1.cuda(), fake.cuda()
        except:
            skipIter = True
        #if (startPoint//1000) > 2500:
        #    skipIter = True
        #skipIter = True
        #t2 = t1
        #t1 = time.time()-1562013000.0
        #diff = t1 - t2
        #print (diff)
        #if diff > 0.2:
        #    print (diff)
        if not skipIter:
            if updateModel:
                optimizer1.zero_grad()
            #print (a)
            for b in range(0, latentIters):
                #print ("b")
                lastOne = (b == latentIters - 1)
                code = con(fake)
                #'''
                if ((a == 0) and (b == 0)) and oldCodeTest:
                    oldCode = code

                #print (code[0].data.numpy())
                if lastOne:
                    codeNoise = normal.rsample(sample_shape=code.shape).view(code.shape) * 0.01
                    if doCuda:
                        codeNoise = codeNoise.cuda()
                    code = code + codeNoise

                if latentIters == 1:
                    code = code * 0.0
                    if oldCodeTest:
                        oldCode = oldCode * 0.0
                #code[:, 5:] = 0.0
                #'''
                #code[:, 0] = torch.tensor(codes2[0][a][0]).float().cuda()
                #code[:, 1] = torch.tensor(codes2[0][a][1]).float().cuda()
                #code[:, 2] = torch.tensor(codes2[0][a][2]).float().cuda()
                #code[:, 3] = torch.tensor(codes2[0][a][3]).float().cuda()
                #code[:, 4] = torch.tensor(codes2[0][a][4]).float().cuda()

                if lastOne and oldCodeTest:
                    loss = model(x1, y1, oldCode)
                else:
                    loss = model(x1, y1, code)
                #print (time.time()-1562013000.0)
                loss1 = loss.cpu().data.numpy()
                loss = loss + ((torch.mean(code**2.0) * 10.0) * 0.001)
                #print (a)
                #print (loss)
                if not lastOne:
                    optimizer2.zero_grad()
                if (not lastOne) or updateModel:
                    loss.backward()
                if not lastOne:
                    nn.utils.clip_grad_norm_(con.parameters(), max_norm=1.0)
                    optimizer2.step()

                #quit()

            losses.append(loss1)
            if updateModel:
                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
                optimizer1.step()
            oldCode = code.clone()
            codes.append(oldCode[0].cpu().data.numpy())
        #quit()
        #'''
        if (a+1)%200==0:
            len2 = len(losses)
            #quit()
            print (name)
            print (a)
            modelName = './Models/ARM4/' + name + ".pt"
            torch.save(model, modelName)
            #quit()
            #print (loss1)
            if a > 200:
                print (np.mean(losses[-200:]))
            lossName = "./Results/" + name + "_Losses.npy"
            np.save(lossName, losses)
            #codeName = "./InputResults/" + name + "_10B2_TestCodes.npy"
            #np.save(codeName, codes)
            #np.save("./Results/codes2ARM7.npy", codes)
            if len1 == len2:
                print (len1)
                print (len2)
                quit()
            len1 = len(losses)

#print (np.load('./InputResults/ARMBatch10_codes.npy').shape)
#print (np.load('InputResults/CodesPred_M2.npy').shape)
#quit()
#codeBatchPredictor()
#quit()

#trainLatentARM(updateModel=False, latentIters=1, randomStartPoints=False, doCuda=True, name="M2")
#quit()

'''
losses1 = np.load("./Results/M1_TestLosses.npy")
losses2 = np.load("./Results/M2_TestLosses.npy")
losses3 = np.load("./Results/M2_10B2_TestLosses.npy")
losses4 = np.load("./Results/M2_10B3_2_TestLosses.npy")

#print (losses3.shape)
#print (losses4.shape)

len1 = losses3.shape[0] // 100
losses3 = movingAverage(losses3, 100, "mean")[0::100]
losses4 = movingAverage(losses4, 100, "mean")[0::100]
#plt.plot(np.arange(len1)*100, losses1[:len1])
#plt.plot(losses1[1000:1500])
#plt.plot(losses2[1000:1500])
#diff = losses1[:len1-1] - losses3
#plt.hist(diff, bins=100)
#plt.show()
#quit()
print (len1)
plt.plot(losses1[1700:1700+len1])
plt.plot(losses3[:len1])
#plt.plot(losses4[:len1])
plt.show()
quit()
#'''

#doPlots()
#quit()
#print (torch.cuda.is_available())
#quit()

try:
    trainLatentARM(updateModel=True, latentIters=1, randomStartPoints=True, doCuda=True, name="Z1")
except:
    print ("First Quit")
trainLatentARM(updateModel=True, latentIters=5, randomStartPoints=True, doCuda=True, name="Z2")
quit()
#preProcessing()
#quit()
#trainARM()
#quit()
