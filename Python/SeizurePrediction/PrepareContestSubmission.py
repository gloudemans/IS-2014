import os
import random
import scipy.io # for loadmat
import scipy.signal # for detrend
import numpy
import pickle
import SequenceDecimatingNetwork

# Arrange the .mat format data files distributed by Kaggle into the following
# directory structure.
#
# * sRoot/Dog_1
# * sRoot/Dog_2
# * sRoot/Dog_3
# * sRoot/Dog_4
# * sRoot/Dog_5
# * sRoot/Patient_1
# * sRoot/Patient_2

def Go(sDatasetPath, rSampleFrequency, tlGeometry, rHoldout=0.2):

    # List the patients
    lPatients = [('Dog_1',16)] # ,('Dog_2',16),('Dog_3',16),('Dog_4',16),('Dog_5',15),('Patient_1',15),('Patient_2',24)]

    # For each patient...
    for (sPatient,iElectrodes) in lPatients:

        # Path to patient data
        sPatientPath = os.path.join(sDatasetPath,sPatient)

        # Create shuffle file if necessary
        sShufflePath = CreateSplits(sPatientPath)

        # Path to sample rate directory
        sRatePath = os.path.join(sPatientPath, "{:d}Hz".format(round(rSampleFrequency)))

        # If the rate specific directory doesn't exist...
        if not os.path.isdir(sRatePath):

            # Make it
            os.makedirs(sRatePath)

        # Preprocess files to desired sample rate if necesary 
        PreprocessMatFiles(sPatientPath, sRatePath, rSampleFrequency)

        # Train autoencoder using the specified geometry
        # TrainDecimatingAutoencoder(sRatePath)

        TrainClassifier(sShufflePath, sRatePath, iElectrodes, tlGeometry, rHoldout)

# Get name for the model
def GetModelName(sRatePath, iElectrodes, tlGeometry):

    sName = "Model";
    iFeatures = iElectrodes
    for (a,b) in tlGeometry:
        sName += "_{:d}x{:d}".format(a*iFeatures,b)
        iFeatures = b
    sName += ".pkl"

    return(os.path.join(sRatePath,sName))

def TrainClassifier(sShufflePath, sRatePath, iElectrodes, tlGeometry, rHoldout=0.2, iMaxFiles=100):

    rWeightInitScale = 0.01

    def LoadTrainingShuffle(sSrc):

        f = open(sSrc,'rt')
        l = f.read().split('\n')
        f.close()

        lTest   = [s for s in l if('test'       in s)]  
        lTrain1 = [s for s in l if('preictal'   in s)]
        lTrain0 = [s for s in l if('interictal' in s)]

        return(lTrain0, lTrain1, lTest)

    sModelName = GetModelName(sRatePath, iElectrodes, tlGeometry)

    if(os.path.exists(sModelName)):

        f = open(sModelName,"rb")
        oModel = pickle.load(f)
        f.close()

    else:

        oaLayers = [];
        iV = iElectrodes
        for (iDecimation, iH) in tlGeometry:
            raaW = numpy.random.randn(iDecimation*iV,iH)*rWeightInitScale
            raB  = numpy.zeros(iH)
            oLayer = SequenceDecimatingNetwork.SequenceDecimatingNetwork.Layer(iDecimation, raaW, raB)
            oaLayers.append(oLayer) 

        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

    # Load model if it exists, otherwise create it
    def LoadModel(sRatePath, iElectrodes, tlGeometry):

        sName = GetModelName(sRatePath, iElectrodes, tlGeometry)

        print(sName)

    # Read the shuffle file and use the validation split to partition
    # training files into four groups (preictal,interictal) x (training, validation)
    (lTrain0, lTrain1, lTest) = LoadTrainingShuffle(sShufflePath)

    LoadModel(sRatePath, iElectrodes, tlGeometry)

    # When reading files, read them in list order and wrap to the start when the end
    # is reached.

    # Given an upper bound on the number of files to read into memory at one time
    # Read preictal files until up to half the limit are loaded
    # Read interictal files up to the limit

    # Get names of all the .pkl files
   #lFiles0 = [f for f in os.listdir(sPatientPath) if('interictal' in f)]
    #lFiles1 = [f for f in os.listdir(sPatientPath) if('preictal'   in f)]



def CreateSplits(sPatientPath):

    # Path to Shuffle.csv
    sShufflePath = os.path.join(sPatientPath, "Shuffle.csv")

    # If Shuffle.csv does not exist...
    if(not os.path.exists(sShufflePath)):

        # Get names of all the .mat files
        lMatFiles = [f for f in os.listdir(sPatientPath) if(f.endswith('.mat'))]

        # Shuffle the file names
        random.shuffle(lMatFiles)

        # Open Shuffle.csv
        oFile = open(sShufflePath,'wt')

        # For each filename...
        for s in lMatFiles:

            # Write the filename without extension
            print(s[:-4], file=oFile)

        # Close the file
        oFile.close()

    return(sShufflePath)

def PreprocessMatFiles(sSrc, sDst, rSampleFrequency=400, bDetrend=True):

    ## 
    # Load the .mat file specified by sFile and return the
    # data array, the number of electrodes, the number of samples,
    # the length, and the sampling frequency as a tuple.

    def LoadMat(sFile):

        # Load one file
        oMat = scipy.io.loadmat(sFile)

        # Find the variable name
        lKeys = [s for s in oMat.keys() if "segment" in s]

        raaData     = oMat[lKeys[0]]['data'][0,0]
        iElectrodes = oMat[lKeys[0]]['data'][0,0].shape[0]
        iSamples    = oMat[lKeys[0]]['data'][0,0].shape[1]
        rLength     = oMat[lKeys[0]]['data_length_sec'][0,0][0,0]   
        rFrequency  = oMat[lKeys[0]]['sampling_frequency'][0,0][0,0]

        return((raaData, iElectrodes, iSamples, rLength, rFrequency))

    def ClassFromName(sFile):

        # Strip any leading path and force to lowercase
        sName = os.path.basename(sFile).lower()

        # If the name contains the string test...
        if 'test' in sName:
            sClass = 'Test'

        elif 'preictal' in sName:
            sClass = 'Preictal'

        else:
            sClass = 'Interictal'

        return(sClass)

    # Always normalize the data
    bNormalize = True

    # Always use the same peak to average signal ratio
    rPeakAverageRatio = 12

    # Get all mat filenames
    lFiles = [f for f in os.listdir(sSrc) if(f.endswith('.mat'))]

    # For every matfile...
    for iFile in range(len(lFiles)):

        # Get file name
        f = lFiles[iFile]

        # Construct the pickle filename
        sDstFile = sDst + '\\' + f[:-3] + 'pkl'

        # Determine the sample class
        sClass = ClassFromName(f)

        # If the output file doesn't exist
        if(not os.path.exists(sDstFile)):

            # Load the matfile
            (raaData, iElectrodes, iSamples, rLength, rFrequency) = LoadMat(sSrc + '\\' + f)

            # Compute the nearest integer decimation ratio
            iDecimationRatio = int(round(rFrequency/rSampleFrequency))

            # If detrending...
            if bDetrend:

                # Detrend along time axis
                raaData = scipy.signal.detrend(raaData, axis=1).astype(numpy.float32)

            # If decimating...
            if iDecimationRatio > 1:

                # Decimate using 8th order chebyshev IIR
                raaData = scipy.signal.decimate(raaData, iDecimationRatio, axis=1).astype(numpy.float32)

            # If normalizing...
            if bNormalize:

                # For each electrode...
                for iElectrode in range(iElectrodes):

                    # Force unit standard deviation
                    raaData[iElectrode,:] /= raaData[iElectrode,:].std()

                # Scale to specified peak to average ratio
                raaData = numpy.maximum(numpy.minimum(1,raaData/(rPeakAverageRatio)),-1).astype(numpy.float32)

                # Transform between zero and one
                raaData = (raaData+1)/2

            # Pickle a tuple with fields we want
            pickle.dump((raaData.T, rFrequency, sClass), open(sDstFile,'wb'))

        print('{:4d} of {:4d} {}'.format(iFile,len(lFiles),f))

Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(16,128),(8,128),(8,128),(8,1)],.2)



# train network(validation fraction, decimation ratios, layer sizes)
#     for each individual...
#         train a network(validation fraction)
#         assess network(validation fraction)
#         compute test sample predictions()

#     combine test sample predictions
#     upload test sample predictions

# assess network()
#     apply network(training samples)
#     apply network(validation samples)
#     compute roc curve(training sample)
#     compute roc curve(validation sample)
#     plot roc curves
#     compute area under roc curves

# compute roc curve(samples)
#     load validation samples and class targets
#     compute validation predictions
#     sort predictions & class targets by prediction
#     for each prediction score compute false positive rate and false negative rate

# compute validation auc(training fraction)
#     plot curve connecting [0,0] [fp0,fn0], [fp1,fn1],... [1,1]
#     compute sum of areas under each line segment (fp1+fp0)*(fn1-fn0)/2

# compute test sample predictions(max files)
#     while test files remain
#         load up to max files test files
#             for each file
#                 for each offset
#                     add a row
#             compute all activations
#             for each file
#                 average all offsets to form prediction

# def LoadTraining(sSrc, rFraction=0.80, iMaxFiles=None):





#     def LoadTrainingShuffle(sSrc):

#         f = open(sSrc,'rt')
#         l = f.read().split('\n')
#         f.close()

#         lTest   = [s[:-4] for s in l if('test'       in s)]  
#         lTrain1 = [s[:-4] for s in l if('preictal'   in s)]
#         lTrain0 = [s[:-4] for s in l if('interictal' in s)]

#         return(lTrain0, lTrain1, lTest)

#     def SplitTraining(sSrc, rFraction=0.80, iMaxFiles=None):

#         # Get the shuffled filenames 
#         (lTrain0, lTrain1, lTest) = LoadTrainingShuffle(sSrc)

#         # Number of available training files
#         iTrainingFiles = len(lTrain0)+len(lTrain1)

#         # If max files is specified...
#         if(iMaxFiles):

#             # Cap the number of training files
#             iTrainingFiles = min(iTrainingFiles,iMaxFiles)

#         # Load as many preictal files as possible without exceeding half of the total training files
#         iTrain1 = min(len(lTrain1), math.ceil(iTrainingFiles/2))

#         # Load as many interictal files as possible without exceeding the total training files
#         iTrain0 = min(len(lTrain0), iTrainingFiles-iTrain1)

#         # Class 0 training files
#         iTrain0T = round(iTrain0*rFraction)

#         # Class 0 validation files
#         iTrain0V = iTrain0-iTrain0T

#         # CLass 1 training files
#         iTrain1T = round(iTrain1*rFraction)

#         # Class 1 validation files
#         iTrain1V = iTrain1-iTrain1T

#         # Form the set of all files to use for training
#         lT0 = lTrain0[:iTrain0T]
#         lT1 = lTrain1[:iTrain1T]

#         # Form the set of all files to use for validation
#         lV0 = lTrain0[iTrain0T:(iTrain0T+iTrain0V)]
#         lV1 = lTrain1[iTrain1T:(iTrain1T+iTrain1V)]

#         return(lT0, lT1, lV0, lV1)

#     def LoadFiles(sSrc, lFiles):

#         raaaData = None;

#         iFiles = len(lFiles)
#         for k in range(iFiles):
#             sFile = lFiles[k]+'.pkl'
#             print(sFile)
#             (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sSrc,sFile),'rb') )
#             if(raaaData==None):
#                 raaaData = numpy.empty((iFiles, raaData.shape[0], raaData.shape[1]))
#             raaaData[k,:,:] =  raaData;

#         return(raaaData)

#     (lTrain0, lTrain1, lValid0, lValid1) = SplitTraining(os.path.join(sSrc,'Shuffle.csv'), rFraction,iMaxFiles)

#     raaaTrain0 = LoadFiles(sSrc, lTrain0)
#     raaaTrain1 = LoadFiles(sSrc, lTrain1)
#     raaaValid0 = LoadFiles(sSrc, lValid0)
#     raaaValid1 = LoadFiles(sSrc, lValid1)

#     return(raaaTrain0,raaaTrain1,raaaValid0,raaaValid1)

# def ConstructTrainingVectors(raaa0, raaa1, iPatterns, iLength, lDecimation, oaLayers):

#     iFiles0    = raaa0.shape[0]
#     iFiles1    = raaa1.shape[0]

#     iSamples   = raaa0.shape[1]
#     iFeatures  = raaa0.shape[2]

#     raaTrain = numpy.empty((iPatterns,iFeatures*iLength))
#     iaTrain  = numpy.empty(iPatterns)

#     for k in range(iPatterns):
#         # Alternate between classes
#         iaTrain[k] = k % 2

#         # Choose a random offset within the file
#         iOffset = random.randrange(iSamples-iLength)

#         iDecimation = 16

#         # If preictal...
#         if iaTrain[k]:

#             # Get preictal pattern for layer 0
#             raaX = numpy.reshape(raaa1[random.randrange(iFiles1),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )
        
#         else:       
#             # Get interictal pattern for layer 1            
#             raaX = numpy.reshape(raaa0[random.randrange(iFiles0),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )

#         # Now ra is flat with dimension iLength x iFeatures
#         for iLayer in range(len(oaLayers)):

#             # Compute the layer activation
#             raaA = numpy.dot(raaX,oaLayers[iLayer].raaW) + oaLayers[iLayer].raH

#             # Compute the layer output
#             raaY = 1./(1+numpy.exp(-raaA))

#             iDecimation *= lDecimation[iLayer]

#             raaX = numpy.reshape(raaY, (-1, oaLayers[iLayer].iH*lDecimation[iLayer]) )


#         raaTrain[k,:] = raaX

#     return(raaTrain, iaTrain)


# def FlipFlop(raaa0, raaa1, iPatterns, iLength, lDecimation, oaLayers):

#     iFiles0    = raaa0.shape[0]
#     iFiles1    = raaa1.shape[0]

#     iSamples   = raaa0.shape[1]
#     iFeatures  = raaa0.shape[2]

#     raaTrain = numpy.empty((iPatterns,iFeatures*iLength))
#     iaTrain  = numpy.empty(iPatterns)

#     for k in range(iPatterns):
#         # Alternate between classes
#         iaTrain[k] = k % 2

#         # Choose a random offset within the file
#         iOffset = random.randrange(iSamples-iLength)

#         iDecimation = 16

#         # If preictal...
#         if iaTrain[k]:

#             # Get preictal pattern for layer 0
#             raaX = numpy.reshape(raaa1[random.randrange(iFiles1),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )
        
#         else:       
#             # Get interictal pattern for layer 1            
#             raaX = numpy.reshape(raaa0[random.randrange(iFiles0),iOffset:iOffset+iLength,:], (-1, iFeatures*iDecimation) )

#         raX = raaX[:]

#         # Now ra is flat with dimension iLength x iFeatures
#         for iLayer in range(len(oaLayers)):

#             # Compute the layer activation
#             raaA = numpy.dot(raaX,oaLayers[iLayer].raaW) + oaLayers[iLayer].raH

#             # Compute the layer output
#             raaY = 1./(1+numpy.exp(-raaA))

#             raaX = numpy.reshape(raaY, (-1, oaLayers[iLayer].iH*lDecimation[iLayer]) )

#         # Now ra is flat with dimension iLength x iFeatures
#         for iLayer in range(len(oaLayers)-1,-1,-1):

#             # print(raaX.shape)
#             # print(oaLayers[iLayer].raaW.T.shape)
#             # print(oaLayers[iLayer].raV.shape)

#             # Compute the layer activation
#             raaA = numpy.dot(raaX,oaLayers[iLayer].raaW.T) + oaLayers[iLayer].raV

#             # Compute the layer output
#             raaY = 1./(1+numpy.exp(-raaA))

#             raaX = numpy.reshape(raaY, (-1, oaLayers[iLayer].iH) )

#         rRmse = numpy.std(raaX[:]-raX)

#     return(rRmse)



# def Train():

#     oLog = open('Log.txt','at')

#     def Log(sLine):

#         print(sLine)
#         print(sLine, file=oLog)

#     def fEpochReport(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse):

#         Log('iLayer={}, iEpoch={}, bSample={}, rDropV={}, rDropH={}, rRate={}, rMomentum={}, rRmse={}'.format(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse))

#     # Specify epochs
#     iEpochs = 100

#     # Select target directory
#     sDir = os.path.expanduser('~/IS-2014/Python/SeizurePrediction/Dog_1')

#     # Establish training/validation split (80/20) & cap file loading
#     rFraction = 0.8
#     iMaxFiles = 200

#     # Load data into arrays [files,samples,sensors]
#     (raaaTrain0,raaaTrain1,raaaValid0,raaaValid1) = LoadTraining(sDir,rFraction,iMaxFiles)

#     # Construct 100000 training vectors using 16 contiguous samples from each file with no
#     # further neural network processing

#     (raaTrain0, iaTrain0) = ConstructTrainingVectors(raaaTrain0, raaaTrain1, 80000, 16, [], [])
#     (raaValid0, iaValid0) = ConstructTrainingVectors(raaaValid0, raaaValid1, 20000, 16, [], [])

#     # Create 256 x 256 rbm layer
#     oLayer01 = RbmStack.Layer(raaTrain0.shape[1],256)

#     # Create training options
#     oOptions = RbmStack.Options(iEpochs, fEpochReport=fEpochReport)

#     # Create RbmStack
#     oRbmStack = RbmStack.RbmStack([oLayer01])

#     # Train using the specified options
#     oRbmStack.TrainAutoencoder(raaTrain0, oOptions)

#     # Compute training and test set errors
#     (rTrainRmse0) = oRbmStack.ComputeReconstructionError(raaTrain0)
#     (rTestRmse0)  = oRbmStack.ComputeReconstructionError(raaValid0)

#     # Report performance
#     Log('rTrainStd0=  {:.4f}'.format(numpy.std(raaTrain0[:])))
#     Log('rTestStd0=   {:.4f}'.format(numpy.std(raaValid0[:])))
#     Log('rTrainRmse0= {:.4f}'.format(rTrainRmse0))
#     Log('rTestRmse0=  {:.4f}'.format(rTestRmse0))

#     (raaTrain1, iaTrain1) = ConstructTrainingVectors(raaaTrain0, raaaTrain1, 8000, 256, [16], [oLayer01])
#     (raaValid1, iaValid1) = ConstructTrainingVectors(raaaValid0, raaaValid1, 2000, 256, [16], [oLayer01])

#     # Create 4096 x 256 rbm layer
#     oLayer12 = RbmStack.Layer(4096,256)

#     # Create training options
#     oOptions = RbmStack.Options(iEpochs, fEpochReport=fEpochReport)

#     # Create RbmStack
#     oRbmStack = RbmStack.RbmStack([oLayer12])

#     # Train using the specified options
#     oRbmStack.TrainAutoencoder(raaTrain1, oOptions)

#     # Compute training and test set errors
#     (rTrainRmse1) = oRbmStack.ComputeReconstructionError(raaTrain1)
#     (rTestRmse1)  = oRbmStack.ComputeReconstructionError(raaValid1)

#     # Report performance
#     Log('rTrainStd1=  {:.4f}'.format(numpy.std(raaTrain1[:])))
#     Log('rTestStd1=   {:.4f}'.format(numpy.std(raaValid1[:])))
#     Log('rTrainRmse1= {:.4f}'.format(rTrainRmse1))
#     Log('rTestRmse1=  {:.4f}'.format(rTestRmse1))

#     pickle.dump((oLayer01, oLayer12), open('Layer.pkl','wb'))

#     (oLayer01, oLayer12) = pickle.load(open('Layer.pkl','rb'))

#     # Compute training and test set errors
#     rTrainRmse = FlipFlop(raaaTrain0, raaaTrain1, 8000, 256, [16, 1], [oLayer01, oLayer12])
#     rTestRmse  = FlipFlop(raaaValid0, raaaValid1, 2000, 256, [16, 1], [oLayer01, oLayer12])

#     # Report performance
#     Log('rTrainRmse= {:.4f}'.format(rTrainRmse))
#     Log('rTestRmse=  {:.4f}'.format(rTestRmse))

#     oLog.close()

# Train()



#     # Create L0 training patterns based on   16 samples
#     # Train  256x256 W01 network
#     # Create L1 training patterns based on  256 samples processed by W01 network
#     # Train 4096x256 W12 network
#     # Create L2 training patterns based on 4096 samples processed by W01 and W12 network
#     # Train 4096x256 W23 network



#     # Create 8192 feature samples based on  256 L0 samples transformed to 512 features L1 samples impressions of 16 samples of 16 inputs, each 

#     # Train 256x1024 network for epochs and save weights
#     # Create 256 sample training tiles - each tile 16


# #     # Train layer 1

# # def Log(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError):

# #     print('iLayer={}, iEpoch={}, bSample={}, rDropV={}, rDropH={}, rRate={}, rMomentum={}, rRmse={}, rError={}'.format(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError))

# # # Define a function to exercise the class (crudely!)
# # def Tester(sFile, iSize):

# #     import pickle

# #     # Define classes used for RbmStack interfacing
# #     class Layer:

# #         def __init__(self, iSize, iEpochs, sActivationUp='Logistic'):
# #             self.iSize = iSize
# #             self.raaW = object
# #             self.sActivationUp = sActivationUp
# #             self.sActivationDn = "Logistic"
# #             self.raRmse     = numpy.zeros(iEpochs)
# #             self.raError    = numpy.zeros(iEpochs)

# #     class LayerOptions:

# #         def __init__(self, iEpochs):

# #             self.raDropV    = 0.0*numpy.ones(iEpochs)
# #             self.raDropH    = 0.0*numpy.ones(iEpochs)
# #             self.raMomentum = 0.9*numpy.ones(iEpochs)
# #             self.raMomentum[:5]=0.5;
# #             self.raRate     = 0.1*numpy.ones(iEpochs)
# #             self.baSample   = 0*numpy.ones(iEpochs)
# #             self.raRmse     = numpy.zeros(iEpochs)


# #     class Options:

# #         def __init__(self, iEpochs):

# #             self.iEpochs = iEpochs
# #             self.oaLayer = [LayerOptions(iEpochs), LayerOptions(iEpochs), LayerOptions(iEpochs)]  
# #             self.fEvent = Log

# #     # Specify epochs
# #     iEpochs = 50

    
# #     (raaTrain, iaTrain, raaTest, iaTest) = pickle.load( open( sFile, "rb" ) )
# #     print(raaTrain.shape)

# #     raaX = raaTrain

# #     # Create 784 x 1000 rbm layer
# #     oaLayers = [Layer(raaX.shape[1],iEpochs),Layer(1000,iEpochs), Layer(iSize,iEpochs)]

# #     # Create training options
# #     oOptions = Options(iEpochs)

# #     # Create RbmStack
# #     oRbmStack = RbmStack.RbmStack(oaLayers)

# #     # Train using the specified options
# #     oRbmStack.TrainAutoencoder(raaX, oOptions)

# #     # Compute training and test set errors
# #     (rTrainRmse, rTrainError) = oRbmStack.ComputeReconstructionError(raaTrain)
# #     (rTestRmse,  rTestError)  = oRbmStack.ComputeReconstructionError(raaTest)

# #     #     # Report performance
# #     oFile = open('Log.txt','at')
# #     print('sFile={}, iSize={:d}'.format(sFile,iSize),file=oFile)
# #     print('rTrainRmse= {:.4f}, rTrainError= {:.4f}'.format(rTrainRmse, rTrainError),file=oFile)
# #     print('rTestRmse=  {:.4f}, rTestError=  {:.4f}'.format(rTestRmse,  rTestError),file=oFile)
# #     oFile.close()

# # sFile = "Batch_0000.pkl" 
# # Tester(sFile,10)
# # Tester(sFile,20)
# # Tester(sFile,50)
# # Tester(sFile,100)
# # Tester(sFile,200)
# # Tester(sFile,500)
# # Tester(sFile,1000)
# # Tester(sFile,2000)


# # 