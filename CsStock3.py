#Processes the ITCH data

import numpy as np
import csv
import copy

def StockFrequency():
    #DataProcessed2
    stocks = []
    num = 14
    for a in range(0, num):
        print (a)
        name = './ProcessingData/FullDayCSV/05302019.NASDAQ_ITCH50_' + str(a+1) + '.csv'
        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                stocks.append(row[8])
    uni, counts = np.unique(stocks, return_counts=True)
    args = np.argsort(counts)
    np.save('./ProcessingData2/InitialProcessing/StockCounts.npy', [uni[args], counts[args]])
    #['GOOG' 'C' 'XLK' 'SMH' 'UVXY' 'INTC' 'UDOW' 'DIA' 'AAPL' 'EWZ']
    #['MU' 'SQQQ' 'AMD' 'VXX' 'MSFT' 'TQQQ' 'TVIX' 'IWM' 'QQQ' 'SPY']


def saveData(isTrade, stockID):
    data = []
    if isTrade:
        num = 1
    else:
        num = 14
    for a in range(0, num):
        print (a)
        name = './ProcessingData/FullDayCSV/05302019.NASDAQ_ITCH50_'
        if isTrade:
            name = name + "t"
        name = name + str(a+1)+ '.csv'
        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row[8] == stockID: #"QQQ"
                    data.append(row)

    saveName = './ProcessingData2/InitialProcessing/'
    if isTrade:
        saveName = saveName + stockID + "_t.npy"
    else:
        saveName = saveName + stockID + "_o.npy"
    np.save(saveName, data)

def saveModRefs():
    #np.save('./ProcessingData/modRefs/newValidRefs.npy', newValidRefs)
    num = 18
    for a in range(0, num):
        modRefs = []
        newModRefs = []
        print (a)
        name = './ProcessingData/FullDayCSV/05302019.NASDAQ_ITCH50_m' + str(a+1) + '.csv'
        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            #line_count = 0
            for row in csv_reader:
                modRefs.append(row[5])
                newModRefs.append(row[10])
        name2 = './ProcessingData/modRefs/modRefs' + str(a+1) + '.npy'
        np.save(name2, modRefs)
        name3 = './ProcessingData/modRefs/newModRefs' + str(a+1) + '.npy'
        np.save(name3, newModRefs)

def findQQQRefs(stockID): #Outdated name. Should theoreticaly be findStockRefs
    #modRefs = np.load('./ProcessingData/modRefs.npy')
    #print (modRefs.shape)
    #quit()
    #176460403
    #10000000
    #quit()

    '''
    name3 = './ProcessingData/modRefs/newModRefs' + str(1) + '.npy'
    newModRefs = np.load(name3)
    args, counts = np.unique(newModRefs, return_counts=True)
    print (np.unique(counts))
    quit()
    '''


    data = np.load('./ProcessingData2/InitialProcessing/' + stockID + '_o.npy')
    validRefs = list(data[1:].T[5])
    #print (data[:10])
    #print (np.argwhere('111835' == data.T[5]))
    #quit()
    oldValidRefs = validRefs
    del data
    notDone = True
    #print (len(modRefs))
    fullArgs1 = []
    for a in range(0, 18):
        fullArgs1.append([])
    #quit()
    first = True
    b = 0
    while notDone:
        print ("FULL_Loop")
        len1 = len(validRefs)
        #print (len1)
        #quit()
        newValidRefs = []
        for a in range(0, 18):
            print (a)
            name2 = './ProcessingData/modRefs/modRefs' + str(a+1) + '.npy'
            modRefs = np.load(name2)

            #print (np.array(oldValidRefs).shape)
            #print (np.unique(np.array(oldValidRefs)).shape)
            #quit()
            modRefs = list(modRefs)
            inBools = np.isin(np.array(modRefs), np.array(oldValidRefs))
            argsIn = np.squeeze(np.argwhere(inBools == True))
            #print (modRefs[:10])
            #print (oldValidRefs[:10])
            #quit()

            #print (argsIn[:3])
            #print (np.array(modRefs)[argsIn[:3]])
            #print (inBools[argsIn[0]])
            #print (np.array(modRefs)[argsIn[0]] in np.array(oldValidRefs))
            #print (argsIn.shape)
            #quit()

            #del modRefs
            del inBools
            name3 = './ProcessingData/modRefs/newModRefs' + str(a+1) + '.npy'
            newModRefs = np.load(name3)

            curNew = list(np.unique(np.array(newModRefs)[argsIn]))
            if "NA" in curNew:
                curNew.remove("NA")
            newValidRefs = newValidRefs + curNew
            #if a > 2:
            fullArgs1[a] = fullArgs1[a] + list(argsIn)
            #print (modRefs[argsIn[0]])
            #print (modRefs[argsIn[1]])
            #print (modRefs[argsIn[1]] in oldValidRefs)
            #quit()

        print (len(newValidRefs))
        #print (np.unique(np.array(newValidRefs)).shape)
        validRefs = validRefs + newValidRefs
        oldValidRefs = np.copy(newValidRefs)
        np.save('./ProcessingData2/modRefs/' + stockID + '_newValidRefs.npy', newValidRefs)
        if len1 == len(validRefs):
            notDone = False

        #print (fullArgs1[0][0])
        #newModRefs[fullArgs1[0][0]]
        #quit()
        np.save('./ProcessingData2/modRefs/' + stockID + '_fullArgs1.npy', fullArgs1)

        if len(newValidRefs) < 500:
            quit()
        b += 1

def manyModRefs(stockID):
    ValidRefs = np.load('./ProcessingData2/modRefs/' + stockID + '_newValidRefs.npy')
    ValidRefs = list(ValidRefs)
    fullArgs2 = []
    for a in range(0, 18):
        fullArgs2.append([])
    num = 18
    for a in range(0, num):
        print (a)
        print (len(ValidRefs))
        modRefs = []
        newModRefs = []
        name = './ProcessingData/FullDayCSV/05302019.NASDAQ_ITCH50_m' + str(a+1) + '.csv'
        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                #print (line_count)
                if row[5] in ValidRefs:
                    fullArgs2[a].append(line_count)
                    if row[10] != "NA":
                        ValidRefs.remove(row[5])
                        ValidRefs.append(row[10])
                line_count += 1

        np.save('./ProcessingData2/modRefs/' + stockID + '_fullArgs2.npy', fullArgs2)

def combineFullArgs(stockID):
    #'''
    print ("A")
    fullArgs1 = np.load('./ProcessingData2/modRefs/' + stockID + '_fullArgs1.npy')
    for a in range(0, 18):
        name1 = './ProcessingData2/modRefs/' + stockID + '_fullArgs1_' + str(a+1) + '.npy'
        np.save(name1, fullArgs1[a])
    del fullArgs1
    print ("B")
    fullArgs1 = np.load('./ProcessingData2/modRefs/' + stockID + '_fullArgs2.npy')
    for a in range(0, 18):
        name1 = './ProcessingData2/modRefs/' + stockID + '_fullArgs2_' + str(a+1) + '.npy'
        np.save(name1, fullArgs1[a])
    del fullArgs1
    print ("C")
    #'''


    ValidRefs = np.load('./ProcessingData2/modRefs/' + stockID + '_newValidRefs.npy')
    ValidRefs = list(ValidRefs)
    fullArgs2 = []
    for a in range(0, 18):
        fullArgs2.append([])
    num = 18
    for a in range(0, num):
        print (a)
        name1 = './ProcessingData2/modRefs/' + stockID + '_fullArgs1_' + str(a+1) + '.npy'
        name2 = './ProcessingData2/modRefs/' + stockID + '_fullArgs2_' + str(a+1) + '.npy'
        args = list(np.load(name1)) + list(np.load(name2))
        #print (args[:10])
        name = './ProcessingData/FullDayCSV/05302019.NASDAQ_ITCH50_m' + str(a+1) + '.csv'
        data = []
        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            curArg = 0
            curVal = args[curArg]
            for row in csv_reader:
                #if curArg == 10:
                    #quit()
                if line_count == curVal:
                    #print (row)
                    #print ("Good")
                    #print (row)
                    #if curArg % 100000 == 0:
                    #    print (curArg)
                    data.append(row)
                    curArg += 1
                    curVal = args[curArg]
                line_count += 1
        name3 = './ProcessingData2/LaterData/mod' + stockID + 'Data' + str(a+1) + '.npy'
        np.save(name3, data)

def processSavedQQQ(stockID):
    #'''
    mArgs = np.array([1, 4, 2, 6, 9, 10, 5])
    num = 18
    for a in range(0, num):
        print (a)
        name1 = './ProcessingData2/LaterData/mod' + stockID + 'Data' + str(a+1) + '.npy'
        data = np.load(name1)
        #if a == 0:
        #    data = data[1:]
        #print (data[:3])
        data = data.T[mArgs].T
        #print (data[:3])
        #quit()
        #print (data[0])
        name2 = './ProcessingData2/LaterData/mod' + stockID + 'Data2_' + str(a+1) + '.npy'
        np.save(name2, data)

    oArgs = np.array([1, 4, 6, 7, 9, 5, 2])
    data = np.load('./ProcessingData2/InitialProcessing/' + stockID + "_o.npy")
    np.save('./ProcessingData2/LaterData/order' + stockID + 'Data.npy', data.T[oArgs].T)

    tArgs = np.array([1, 4, 6, 7, 9, 5, 2])
    data = np.load('./ProcessingData2/InitialProcessing/' + stockID + "_t.npy")
    np.save('./ProcessingData2/LaterData/trade' + stockID + 'Data.npy', data.T[tArgs].T)

def timeOrderMessages(stockID):
    mArg1 = 0
    mArg2 = 0
    tArg = 0
    oArg = 0
    data1 = np.load('./ProcessingData2/LaterData/mod' + stockID + 'Data2_' + str(mArg1+1) + '.npy')
    data2 = np.load('./ProcessingData2/LaterData/trade' + stockID + 'Data.npy')
    data3 = np.load('./ProcessingData2/LaterData/order' + stockID + 'Data.npy')

    data = np.load('./ProcessingData2/LaterData/order' + stockID + 'Data.npy')
    mVal = float(data1[mArg2][1])
    tVal = float(data2[tArg][1])
    oVal = float(data3[oArg][1])

    notDone = True
    fileCount = 0
    while notDone:
        print (fileCount)
        #10000000
        data = []
        size = 0
        while size < 10000000:
            #if size % 100000:
            #    print (size)
            #    print(data[:10])
            if (mVal < tVal) and (mVal < oVal):
                best = 0
            elif tVal < oVal:
                best = 1
            else:
                best = 2

            if best == 0:
                data.append(data1[mArg2])
                mArg2 += 1
            if best == 1:
                data.append(data2[tArg])
                tArg += 1
            if best == 2:
                data.append(data3[oArg])
                oArg += 1
            #print (best)
            if (data1.shape[0] <= mArg2) and (mArg1 < 17):
                mArg1 += 1
                mArg2 = 0
                #print ('./ProcessingData/modQQQData2_' + str(mArg1+1) + '.npy')
                data1 = np.load('./ProcessingData2/LaterData/mod' + stockID + 'Data2_' + str(mArg1+1) + '.npy')[1:]
            size += 1
            #print (best)
            infinity1 = 100000000000000000000
            if mArg2 < data1.shape[0]:
                #if mArg2 == 0:
                #    print (mArg1)
                mVal = float(data1[mArg2][1])
            else:
                mVal = infinity1
            if tArg < data2.shape[0]:
                tVal = float(data2[tArg][1])
            else:
                tVal = infinity1
            if oArg < data3.shape[0]:
                oVal = float(data3[oArg][1])
            else:
                oVal = infinity1
            if min(oVal, mVal, tVal) == infinity1:
                notDone = False
                size = infinity1
        #print (oArg)
        print (len(data))
        #np.save('./ProcessingData/AllMessages/messages' + str(fileCount) + ".npy", data)
        #np.save('./ProcessingData/AllMessages/messages.npy', data)
        np.save('./ProcessingData2/AllMessages/' + stockID + '_messages.npy', data)
        fileCount += 1

def SanityCheckMod(stockID):

    data = np.load('./ProcessingData2/AllMessages/' + stockID + '_messages.npy')
    notDone = True
    a = 0
    uCount = 0
    while notDone:
        #'3165443'
        #'9510263'
        if data[a][-2] == '942847':
            print ("Done")
            quit()
        #if data[a][0] == "E":
        #    uCount += 1
        #    if uCount == 1000:
        #        print (data[a])
        #        quit()
        a += 1

def integrateModifications(stockID):
    #data = np.load('./ProcessingData/AllMessages/messages.npy')
    data = np.load('./ProcessingData2/AllMessages/' + stockID + '_messages.npy')
    #print (data[:5])
    #quit()
    refs = data.T[-2]
    size1 = refs.shape[0]
    refsFlip = np.flip(np.copy(refs), axis=0)
    #refs = np.flip(refs)
    len1 = data.shape[0]
    oldB = 50
    #print (data[:5])
    #quit()
    print (len1//1000)
    dataNew = []
    for a in range(0, len1):
        #a = a + (10000*258)
        if a % 10000 == 0:
            print (a//10000)
        if data[a][0] in ['E', 'C', 'X', 'D', 'U']:
            #if (a//10000) == 258:
            #    print (a)
            #print (data[a])
            old_order_ref = data[a][-1]

            b = a
            while (old_order_ref != data[b][-2]) and (a - b) < 20:
                b -= 1
                if b < 0:
                    print ("Error")
            if a - b == 20:
                refs1 = refs[a-100:a]
                answer = np.argwhere(refs1 == old_order_ref)
                if len(list(answer)) != 0:
                    b = answer[0][0] - 100 + a
                else:
                    refs1 = refs[oldB-50:oldB+50]
                    answer = np.argwhere(refs1 == old_order_ref)
                    if len(list(answer)) != 0:
                        b = answer[0][0] - 50 + oldB
                        #print (data[b][-2])
                        #print (old_order_ref)
                        #quit()
                    else:
                        #refs1 = refs[a-1000:a]
                        #answer = np.argwhere(refs1 == old_order_ref)
                        #if len(list(answer)) != 0:
                        #    b = answer[0][0] - 100 + a
                        #else:
                        #    refsFlip
                        refs1 = refsFlip[size1-a:]
                        b = np.argmax(refs1==old_order_ref)#[0][0]
                        #b = np.argwhere(refs == old_order_ref)[0][0]
            #print (a - b)
            oldB = b
            row1 = copy.copy(data[b])
            #print (row1)
            #quit()
            row1[0] = data[a][0]
            row1[1] = data[a][1]
            #if (data[b][0] == "Q") and (row1[2] == "NA"): #This allows one to keep deletion of cross orders
            #    row1[2] = "True"
            if row1[2] != "NA":
                dataNew.append(row1)
            if data[a][0] == 'U':
                row2 = data[a]
                row2[0] = "M"
                dataNew.append(row2)
                #if (a//10000) == 258:
                #    print ("C")
            else:
                data[a][2] = data[b][2]
                data[a][3] = data[b][3]
                data[a][4] = data[b][4]
        else:
            dataNew.append(data[a])

        #if ("NA" == dataNew[-1][2]) and ('Q' != dataNew[-1][0]):
        #    print ("INFO")
        #    print (dataNew[-1])
        #    print (data[b])

    np.save('./ProcessingData2/AllMessages/' + stockID + '_IntegratedMessages.npy', dataNew)

    #print (np.unique(data.T[0]))
    #['A' 'C' 'D' 'E' 'F' 'P' 'Q' 'U' 'X']
    #‘E’, ‘C’, ‘X’, ‘D’, and ‘U’

def numericTranslation(stockID):
    #'289829935'
    #'5261'
    data = np.load('./ProcessingData2/AllMessages/' + stockID + '_IntegratedMessages.npy')
    args = np.squeeze(np.argwhere(data[:, 2] == "NA"))
    #print (data[1000:1010])
    #print (data[args[1]])
    #quit()
    for a in range(0, data.shape[0]):
        if data[a][0] == 'Q':
            if data[a][2] == "NA":
                data[a][2] = 0.0
    #269057
    #print (data[0])
    #print (data[:, 0][np.argwhere(data[:, 2] == "NA")])
    #quit()
    times = data[:, 1]#.astype(float)
    prices = data[:, 4]#.astype(float)
    shares = data[:, 3]#.astype(float)
    messageType = data[:, 0]
    messageTypes = ['A', 'M', 'C', 'D', 'E', 'F', 'P', 'Q', 'U', 'X'] #I added 'M'
    fullOneHot = []
    for a in range(0, len(messageTypes)):
        mType = messageTypes[a]
        currentOneHot = np.zeros(messageType.shape[0])
        currentOneHot[messageType == mType] = 1.0
        fullOneHot.append(currentOneHot)
    Buy = data[:, 2]
    Buy[Buy=="TRUE"] = 1.0
    Buy[Buy=="FALSE"] = 0.0
    newData = [times, prices, shares, Buy]
    newData = np.concatenate((np.array(newData), np.array(fullOneHot)))
    newData = np.array(newData).T
    for a in range(0, newData.shape[0]):
        ar = newData[a]
        if "NA" in ar:
            print ("Error")
            print (a)
            print (ar)
            quit()
    #quit()
    newData = newData.astype(float)
    np.save('./ProcessingData2/FinalData/' + stockID + '_NumericMessages.npy', newData)

def findValidRefs():
    data = np.load('./ProcessingData/QQQ.npy')
    validRefs = list(data[1:].T[5])
    del data
    #10
    data = []
    num = 18
    for a in range(0, num):
        print (a)
        name = './FullDayCSV/05302019.NASDAQ_ITCH50_m' + str(a+1) + '.csv'
        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            #line_count = 0
            for row in csv_reader:
                if row[5] in validRefs:
                    data.append(row)
                    if row[1] == "U":
                        print ("U")
                        validRefs.append(row[10])

    #np.save('./stocks_temp.npy', stocks)
    np.save('./QQQ2.npy', data)

def saveAllData():
    data1 = np.load('./ProcessingData/QQQ.npy')
    print (data1.shape)
    quit()
    data2 = np.load('./QQQ2.npy')
    data3 = np.load('./QQQ3.npy')
    print (data1.shape)
    print (data2.shape)
    print (data3.shape)
    data = np.concatenate([data1.T, data2.T, data3.T])
    print (data.shape)

def fullSaving(stockID):
    print (stockID)
    saveData(False, stockID)
    saveData(True, stockID)
    try:
        findQQQRefs(stockID) #One can manualy choose a cut off or uncomment the 500 cut off
    except:
        "hi"
    manyModRefs(stockID)
    combineFullArgs(stockID)
    processSavedQQQ(stockID)
    timeOrderMessages(stockID)
    integrateModifications(stockID)
    numericTranslation(stockID)


#fullSaving('UVXY')
#fullSaving('SMH')
#fullSaving('XLK')

fullSaving('GOOG')
fullSaving('C')
#integrateModifications('MSFT')


#['', 'msg_type', 'locate_code', 'tracking_number', 'timestamp', 'order_ref', 'buy', 'shares', 'stock', 'price', 'mpid', 'date', 'datetime']
#['msg_type', 'timestamp', 'buy', 'order_ref', 'old_order_ref', 'shares', 'price']



#['', 'msg_type', 'locate_code', 'tracking_number', 'timestamp', 'order_ref', 'buy', 'shares', 'stock', 'price', 'match_number', 'cross_type', 'date', 'datetime']
#['msg_type', 'timestamp', 'buy', 'order_ref', 'old_order_ref', 'shares', 'price']

#['msg_type' 'timestamp' 'buy' 'shares' 'price' 'order_ref' 'new_order_ref']


#['', 'msg_type', 'locate_code', 'tracking_number', 'timestamp', 'order_ref', 'shares', 'match_number', 'printable', 'price', 'new_order_ref', 'date', 'datetime']
#['msg_type', 'timestamp', 'buy', 'order_ref', 'old_order_ref', 'shares', 'price']
