import os
import random
import pickle
import math
import numpy
import RbmStack

def LoadTraining(sSrc, rFraction=0.80, iMaxFiles=None):

    def LoadTrainingShuffle(sSrc):

        f = open(sSrc,'rt')
        l = f.read().split('\n')
        f.close()

        lTest   = [s[:-4] for s in l if('test'       in s)]  
        lTrain1 = [s[:-4] for s in l if('preictal'   in s)]
        lTrain0 = [s[:-4] for s in l if('interictal' in s)]

        return(lTrain0, lTrain1, lTest)

    def SplitTraining(sSrc, rFraction=0.80, iMaxFiles=None):

        # Get the shuffled filenames 
        (lTrain0, lTrain1, lTest) = LoadTrainingShuffle(sSrc)

        # Number of available training files
        iTrainingFiles = len(lTrain0)+len(lTrain1)

        # If max files is specified...
        if(iMaxFiles):

            # Cap the number of training files
            iTrainingFiles = min(iTrainingFiles,iMaxFiles)

        # Load as many preictal files as possible without exceeding half of the total training files
        iTrain1 = min(len(lTrain1), math.ceil(iTrainingFiles/2))

        # Load as many interictal files as possible without exceeding the total training files
        iTrain0 = min(len(lTrain0), iTrainingFiles-iTrain1)

        # Class 0 training files
        iTrain0T = round(iTrain0*rFraction)

        # Class 0 validation files
        iTrain0V = iTrain0-iTrain0T

        # CLass 1 training files
        iTrain1T = round(iTrain1*rFraction)

        # Class 1 validation files
        iTrain1V = iTrain1-iTrain1T

        # Form the set of all files to use for training
        lT0 = lTrain0[:iTrain0T]
        lT1 = lTrain1[:iTrain1T]

        # Form the set of all files to use for validation
        lV0 = lTrain0[iTrain0T:(iTrain0T+iTrain0V)]
        lV1 = lTrain1[iTrain1T:(iTrain1T+iTrain1V)]

        return(lT0, lT1, lV0, lV1)

    def LoadFiles(sSrc, lFiles):

        raaaData = None;

        iFiles = len(lFiles)
        for k in range(iFiles):
            sFile = lFiles[k]+'.pkl'
            print(sFile)
            (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sSrc,sFile),'rb') )
            if(raaaData==None):
                raaaData = numpy.empty((iFiles, raaData.shape[0], raaData.shape[1]))
            raaaData[k,:,:] =  raaData;

        return(raaaData)

    (lTrain0, lTrain1, lValid0, lValid1) = SplitTraining(os.path.join(sSrc,'Shuffle.csv'), rFraction,iMaxFiles)

    raaaTrain0 = LoadFiles(sSrc, lTrain0)
    raaaTrain1 = LoadFiles(sSrc, lTrain1)
    raaaValid0 = LoadFiles(sSrc, lValid0)
    raaaValid1 = LoadFiles(sSrc, lValid1)

    return(raaaTrain0,raaaTrain1,raaaValid0,raaaValid1)

def ConstructTrainingVectors(raaa0, raaa1, iPatterns, iLength, lDecimation, oaLayers):

    iFiles0    = raaa0.shape[0]
    iFiles1    = raaa1.shape[0]

    iSamples   = raaa0.shape[1]
    iFeatures  = raaa0.shape[2]

    raaTrain = numpy.empty((iPatterns,iFeatures*iLength))
    iaTrain  = numpy.empty(iPatterns)

    for k in range(iPatterns):
        # Alternate between classes
        iaTrain[k] = k % 2

        # Choose a random offset within the file
        iOffset = random.randrange(iSamples-iLength)

        iDecimation = 16

        # If preictal...
        if iaTrain[k]:

            # Get preictal pattern for layer 0
            raaX = numpy.reshape(raaa1[random.randrange(iFiles1),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )
        
        else:       
            # Get interictal pattern for layer 1            
            raaX = numpy.reshape(raaa0[random.randrange(iFiles0),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )

        # Now ra is flat with dimension iLength x iFeatures
        for iLayer in range(len(oaLayers)):

            # Compute the layer activation
            raaA = numpy.dot(raaX,oaLayers[iLayer].raaW) + oaLayers[iLayer].raH

            # Compute the layer output
            raaY = 1./(1+numpy.exp(-raaA))

            iDecimation *= lDecimation[iLayer]

            raaX = numpy.reshape(raaY, (-1, oaLayers[iLayer].iH*lDecimation[iLayer]) )


        raaTrain[k,:] = raaX

    return(raaTrain, iaTrain)


def FlipFlop(raaa0, raaa1, iPatterns, iLength, lDecimation, oaLayers):

    iFiles0    = raaa0.shape[0]
    iFiles1    = raaa1.shape[0]

    iSamples   = raaa0.shape[1]
    iFeatures  = raaa0.shape[2]

    raaTrain = numpy.empty((iPatterns,iFeatures*iLength))
    iaTrain  = numpy.empty(iPatterns)

    for k in range(iPatterns):
        # Alternate between classes
        iaTrain[k] = k % 2

        # Choose a random offset within the file
        iOffset = random.randrange(iSamples-iLength)

        iDecimation = 16

        # If preictal...
        if iaTrain[k]:

            # Get preictal pattern for layer 0
            raaX = numpy.reshape(raaa1[random.randrange(iFiles1),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )
        
        else:       
            # Get interictal pattern for layer 1            
            raaX = numpy.reshape(raaa0[random.randrange(iFiles0),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )

        raX = raaX[:]

        # Now ra is flat with dimension iLength x iFeatures
        for iLayer in range(len(oaLayers)):

            # Compute the layer activation
            raaA = numpy.dot(raaX,oaLayers[iLayer].raaW) + oaLayers[iLayer].raH

            # Compute the layer output
            raaY = 1./(1+numpy.exp(-raaA))

            raaX = numpy.reshape(raaY, (-1, oaLayers[iLayer].iH*lDecimation[iLayer]) )

        # Now ra is flat with dimension iLength x iFeatures
        for iLayer in range(len(oaLayers)-1,-1,-1):

            # print(raaX.shape)
            # print(oaLayers[iLayer].raaW.T.shape)
            # print(oaLayers[iLayer].raV.shape)

            # Compute the layer activation
            raaA = numpy.dot(raaX,oaLayers[iLayer].raaW.T) + oaLayers[iLayer].raV

            # Compute the layer output
            raaY = 1./(1+numpy.exp(-raaA))

            raaX = numpy.reshape(raaY, (-1, oaLayers[iLayer].iH) )

        rRmse = numpy.std(raaX[:]-raX)

    return(rRmse)



def Train():

    oLog = open('Log.txt','at')

    def Log(sLine):

        print(sLine)
        print(sLine, file=oLog)

    def fEpochReport(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse):

        Log('iLayer={}, iEpoch={}, bSample={}, rDropV={}, rDropH={}, rRate={}, rMomentum={}, rRmse={}'.format(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse))

    # Specify epochs
    iEpochs = 100

    # Select target directory
    sDir = os.path.expanduser('~/IS-2014/Python/SeizurePrediction/Dog_1')

    # Establish training/validation split (80/20) & cap file loading
    rFraction = 0.8
    iMaxFiles = 100

    # Load data into arrays [files,samples,sensors]
    (raaaTrain0,raaaTrain1,raaaValid0,raaaValid1) = LoadTraining(sDir,rFraction,iMaxFiles)

    # Construct 100000 training vectors using 16 contiguous samples from each file with no
    # further neural network processing

    (raaTrain0, iaTrain0) = ConstructTrainingVectors(raaaTrain0, raaaTrain1, 80000, 16, [], [])
    (raaValid0, iaValid0) = ConstructTrainingVectors(raaaValid0, raaaValid1, 20000, 16, [], [])

    # Create 256 x 256 rbm layer
    oLayer01 = RbmStack.Layer(raaTrain0.shape[1],256)

    # Create training options
    oOptions = RbmStack.Options(iEpochs, fEpochReport=fEpochReport)

    # Create RbmStack
    oRbmStack = RbmStack.RbmStack([oLayer01])

    # Train using the specified options
    oRbmStack.TrainAutoencoder(raaTrain0, oOptions)

    # Compute training and test set errors
    (rTrainRmse0) = oRbmStack.ComputeReconstructionError(raaTrain0)
    (rTestRmse0)  = oRbmStack.ComputeReconstructionError(raaValid0)

    # Report performance
    Log('rTrainStd0=  {:.4f}'.format(numpy.std(raaTrain0[:])))
    Log('rTestStd0=   {:.4f}'.format(numpy.std(raaValid0[:])))
    Log('rTrainRmse0= {:.4f}'.format(rTrainRmse0))
    Log('rTestRmse0=  {:.4f}'.format(rTestRmse0))

    (raaTrain1, iaTrain1) = ConstructTrainingVectors(raaaTrain0, raaaTrain1, 8000, 256, [16], [oLayer01])
    (raaValid1, iaValid1) = ConstructTrainingVectors(raaaValid0, raaaValid1, 2000, 256, [16], [oLayer01])

    # Create 4096 x 256 rbm layer
    oLayer12 = RbmStack.Layer(4096,256)

    # Create training options
    oOptions = RbmStack.Options(iEpochs, fEpochReport=fEpochReport)

    # Create RbmStack
    oRbmStack = RbmStack.RbmStack([oLayer12])

    # Train using the specified options
    oRbmStack.TrainAutoencoder(raaTrain1, oOptions)

    # Compute training and test set errors
    (rTrainRmse1) = oRbmStack.ComputeReconstructionError(raaTrain1)
    (rTestRmse1)  = oRbmStack.ComputeReconstructionError(raaValid1)

    # Report performance
    Log('rTrainStd1=  {:.4f}'.format(numpy.std(raaTrain1[:])))
    Log('rTestStd1=   {:.4f}'.format(numpy.std(raaValid1[:])))
    Log('rTrainRmse1= {:.4f}'.format(rTrainRmse1))
    Log('rTestRmse1=  {:.4f}'.format(rTestRmse1))

    pickle.dump((oLayer01, oLayer12), open('Layer.pkl','wb'))

    (oLayer01, oLayer12) = pickle.load(open('Layer.pkl','rb'))

    # Compute training and test set errors
    rTrainRmse = FlipFlop(raaaTrain0, raaaTrain1, 8000, 256, [16, 1], [oLayer01, oLayer12])
    rTestRmse  = FlipFlop(raaaValid0, raaaValid1, 2000, 256, [16, 1], [oLayer01, oLayer12])

    # Report performance
    Log('rTrainRmse= {:.4f}'.format(rTrainRmse))
    Log('rTestRmse=  {:.4f}'.format(rTestRmse))

    oLog.close()

Train()



    # Create L0 training patterns based on   16 samples
    # Train  256x256 W01 network
    # Create L1 training patterns based on  256 samples processed by W01 network
    # Train 4096x256 W12 network
    # Create L2 training patterns based on 4096 samples processed by W01 and W12 network
    # Train 4096x256 W23 network



    # Create 8192 feature samples based on  256 L0 samples transformed to 512 features L1 samples impressions of 16 samples of 16 inputs, each 

    # Train 256x1024 network for epochs and save weights
    # Create 256 sample training tiles - each tile 16


#     # Train layer 1

# def Log(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError):

#     print('iLayer={}, iEpoch={}, bSample={}, rDropV={}, rDropH={}, rRate={}, rMomentum={}, rRmse={}, rError={}'.format(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError))

# # Define a function to exercise the class (crudely!)
# def Tester(sFile, iSize):

#     import pickle

#     # Define classes used for RbmStack interfacing
#     class Layer:

#         def __init__(self, iSize, iEpochs, sActivationUp='Logistic'):
#             self.iSize = iSize
#             self.raaW = object
#             self.sActivationUp = sActivationUp
#             self.sActivationDn = "Logistic"
#             self.raRmse     = numpy.zeros(iEpochs)
#             self.raError    = numpy.zeros(iEpochs)

#     class LayerOptions:

#         def __init__(self, iEpochs):

#             self.raDropV    = 0.0*numpy.ones(iEpochs)
#             self.raDropH    = 0.0*numpy.ones(iEpochs)
#             self.raMomentum = 0.9*numpy.ones(iEpochs)
#             self.raMomentum[:5]=0.5;
#             self.raRate     = 0.1*numpy.ones(iEpochs)
#             self.baSample   = 0*numpy.ones(iEpochs)
#             self.raRmse     = numpy.zeros(iEpochs)


#     class Options:

#         def __init__(self, iEpochs):

#             self.iEpochs = iEpochs
#             self.oaLayer = [LayerOptions(iEpochs), LayerOptions(iEpochs), LayerOptions(iEpochs)]  
#             self.fEvent = Log

#     # Specify epochs
#     iEpochs = 50

    
#     (raaTrain, iaTrain, raaTest, iaTest) = pickle.load( open( sFile, "rb" ) )
#     print(raaTrain.shape)

#     raaX = raaTrain

#     # Create 784 x 1000 rbm layer
#     oaLayers = [Layer(raaX.shape[1],iEpochs),Layer(1000,iEpochs), Layer(iSize,iEpochs)]

#     # Create training options
#     oOptions = Options(iEpochs)

#     # Create RbmStack
#     oRbmStack = RbmStack.RbmStack(oaLayers)

#     # Train using the specified options
#     oRbmStack.TrainAutoencoder(raaX, oOptions)

#     # Compute training and test set errors
#     (rTrainRmse, rTrainError) = oRbmStack.ComputeReconstructionError(raaTrain)
#     (rTestRmse,  rTestError)  = oRbmStack.ComputeReconstructionError(raaTest)

#     #     # Report performance
#     oFile = open('Log.txt','at')
#     print('sFile={}, iSize={:d}'.format(sFile,iSize),file=oFile)
#     print('rTrainRmse= {:.4f}, rTrainError= {:.4f}'.format(rTrainRmse, rTrainError),file=oFile)
#     print('rTestRmse=  {:.4f}, rTestError=  {:.4f}'.format(rTestRmse,  rTestError),file=oFile)
#     oFile.close()

# sFile = "Batch_0000.pkl" 
# Tester(sFile,10)
# Tester(sFile,20)
# Tester(sFile,50)
# Tester(sFile,100)
# Tester(sFile,200)
# Tester(sFile,500)
# Tester(sFile,1000)
# Tester(sFile,2000)


