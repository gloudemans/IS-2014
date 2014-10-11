import os
import random
#

def SetTrainingShuffle(sSrc, sDst):

     # Get all labeled samples

    lTest   = [f for f in os.listdir(sSrc) if('test'       in f)]  
    lTrain1 = [f for f in os.listdir(sSrc) if('preictal'   in f)]
    lTrain0 = [f for f in os.listdir(sSrc) if('interictal' in f)]

    lTest.sort()
    random.shuffle(lTrain1)
    random.shuffle(lTrain0)

    #lTest   = [s[:-4] for s in lTest  ]
    #lTrain0 = [s[:-4] for s in lTrain0]
    #lTrain1 = [s[:-4] for s in lTrain1]

    oFile = open(sDst,'wt')

    for s in lTrain0:
        print(s, file=oFile)

    for s in lTrain1:
        print(s, file=oFile)

    for s in lTest:
        print(s, file=oFile)

    oFile.close()

def LoadTrainingShuffle(sSrc):

    f = open(sSrc,'rt')
    l = f.read().split('\n')
    f.close()

    lTest   = [s[:-4] for s in l if('test'       in s)]  
    lTrain1 = [s[:-4] for s in l if('preictal'   in s)]
    lTrain0 = [s[:-4] for s in l if('interictal' in s)]

    return(lTrain0, lTrain1, lTest)

def ShuffleAll(sSrc):

    # Get subdirectories of the specified source directory
    lSubs = [os.path.join(sSrc,f) for f in os.listdir(sSrc) if os.path.isdir(os.path.join(sSrc, f))]

    # For each subdirectory...
    for s in lSubs:

        # Shuffle training names
        SetTrainingShuffle(s, os.path.join(s,'Shuffle.csv')) 

# sPath='C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw'
# ShuffleAll(sPath)
# (lTrain0, lTrain1, lTest) = LoadTrainingShuffle(os.path.join(sPath,'Dog_1\\Shuffle.csv'));



