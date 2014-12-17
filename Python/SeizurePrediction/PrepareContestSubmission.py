import os
import random
import scipy.io # for loadmat
import scipy.signal # for detrend
import numpy
import pickle
import RbmStack
import SequenceDecimatingNetwork

import matplotlib.pyplot as plt

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

def Go(sDatasetPath, rSampleFrequency, tlGeometry, rHoldout=0.2, bRetrain=False):

    # List (patients directory name, electrode count) 
    lPatients = [('Dog_1',16),('Dog_2',16),('Dog_3',16),('Dog_4',16),('Dog_5',15),('Patient_1',15),('Patient_2',24)]

    # For each patient...
    for (sPatient,iSensors) in lPatients:

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

        # Train autoencoder using the specified geometry if necessary
        TrainDecimatingAutoencoder(sShufflePath, sRatePath, iSensors, tlGeometry, rHoldout, bRetrain)

        # Train classifier using the specified geometry
        TrainClassifier(sShufflePath, sRatePath, iSensors, tlGeometry, rHoldout)

    # Read file into list
    fOut = open(os.path.join(sDatasetPath,"Upload.csv"),'wt')
    fOut.write("clip,preictal\n".format())

    # For each patient...
    for (sPatient,iSensors) in lPatients:

        # Path to patient data
        sPatientPath = os.path.join(sDatasetPath,sPatient)

        # Path to sample rate directory
        sRatePath = os.path.join(sPatientPath, "{:d}Hz".format(round(rSampleFrequency)),"Test.csv")

        # Read file into list
        f = open(sRatePath,'rt')
        l = f.read()
        f.close()

        # Write to upload file
        fOut.write(l)

    fOut.close()

def GenerateTrainingPatterns(sRatePath, iPatterns, iTotalDecimation, iSensors, lT0, lT1, oaLayers):

    iT0 = round(iPatterns/2)
    iT1 = iPatterns - iT0

    # Create an array to hold the output patterns
    raaaX = numpy.zeros((iPatterns, iTotalDecimation, iSensors))

    # Get total file count
    iFiles = len(lT0)+len(lT1)

    ia = numpy.random.permutation(iPatterns)

    iPattern  = 0

    # For each file...
    for k in range(len(lT0)):

        # Get filename
        sFile = lT0[k]+'.pkl'

        # Load the data
        (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sRatePath,sFile),'rb') )

        # Measure the data
        (iSamples, iSensors) = raaData.shape

        # Choose random offsets
        while(iPattern<(k+1)*iT0/len(lT0)):

            # Compute a random offset
            iOffset = random.randrange(iSamples-iTotalDecimation)

            # Save this pattern
            raaaX[ia[iPattern],:,:] = raaData[iOffset:iOffset+iTotalDecimation,:]

            # Next pattern
            iPattern += 1

    # For each file...
    for k in range(len(lT1)):

        # Get filename
        sFile = lT1[k]+'.pkl'

        # Load the data
        (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sRatePath,sFile),'rb') )

        # Measure the data
        (iSamples,iSensors) = raaData.shape

        # Choose random offsets
        while(iPattern<(k+1)*iT1/len(lT1)+iT0):

            # Compute a random offset
            iOffset = random.randrange(iSamples-iTotalDecimation)

            # Save this pattern
            raaaX[ia[iPattern],:,:] = raaData[iOffset:iOffset+iTotalDecimation,:]

            # Next pattern
            iPattern += 1

    oSdn = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

    raaaY = oSdn.ComputeOutputs(raaaX)

    raaaXr = oSdn.ComputeInputs(raaaY)

    rE = numpy.std(raaaX[:]-raaaXr[:])

    print(rE)

    (iP,iSamples,iSensors) = raaaY.shape

    raaY = raaaY.reshape(iP, iSamples*iSensors)

    return(raaY)

def TrainDecimatingAutoencoder(sShufflePath, sRatePath, iSensors, tlGeometry, rHoldout, bRetrain=False):

    iEpochs   =    20
    iPatterns = 20000

     # Get name for the model
    sModelName = GetModelName(sRatePath, iSensors, tlGeometry, "Layers")

    # If the model already exists...
    if(bRetrain or not os.path.exists(sModelName)):

        # Create training options
        oOptions = RbmStack.Options(iEpochs)

        # Load training shuffle
        (lT0, lT1, lV0, lV1, lTest) = LoadTrainingShuffle(sShufflePath, rHoldout)

        # Initialize total decimation ratio
        iTotalDecimation = 1

        # Create a list to store sequence decimation layers
        oaLayers = []; 

        iV = iSensors

        # For each network layer...
        for iLayer in range(len(tlGeometry)):

            # Get decimation and hidden unit count for this layer
            (iDecimation, iH) = tlGeometry[iLayer] 

            # Track total decimation ratio
            iTotalDecimation *= iDecimation

            # Decimate input samples
            iV *= iDecimation

            # Create a single layer Rbm with the correct geometry
            oRbm = RbmStack.RbmStack([RbmStack.Layer(iV,iH)])
            
            # Generate training patterns
            raaX = GenerateTrainingPatterns(sRatePath, iPatterns, iTotalDecimation, iSensors, lT0, lT1, oaLayers)

            print(iPatterns, iTotalDecimation)

            # Compute the standard deviation of training patterns
            rStd = numpy.std(raaX[:])

            print("\nPretraining Layer {:d} with standard deviation {:.4f}\n".format(iLayer, rStd))

            # Train the autoencoder
            oRbm.TrainAutoencoder(raaX, oOptions)

            # Add the new weights
            oaLayers.append(SequenceDecimatingNetwork.Layer(iDecimation, oRbm.oaLayer[0].raaW, oRbm.oaLayer[0].raH, oRbm.oaLayer[0].raV))

            # Hidden outputs are next layer inputs
            iV = iH

        # Generate training patterns
        raaX = GenerateTrainingPatterns(sRatePath, iPatterns, iTotalDecimation, iSensors, lT0, lT1, oaLayers)

        # Save layers
        f = open(sModelName,"wb")
        pickle.dump(oaLayers, f)
        f.close()

        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        # Save model
        f = open(sModelName+"Sdn","wb")
        pickle.dump(oModel, f)
        f.close()

        return(oaLayers)

# Get name for the model
def GetModelName(sRatePath, iSensors, tlGeometry, sName):

    iFeatures = iSensors
    for (a,b) in tlGeometry:
        sName += "_{:d}x{:d}".format(a*iFeatures,b)
        iFeatures = b
    sName += ".pkl"

    return(os.path.join(sRatePath,sName))

def LoadTrainingShuffle(sSrc, rHoldout):

    f = open(sSrc,'rt')
    l = f.read().split('\n')
    f.close()

    # We want to make sure that both the training and validation sets have preictal samples

    lTest   = [s for s in l if('test'       in s)]
    lTrain0 = [s for s in l if('interictal' in s)]
    lTrain1 = [s for s in l if('preictal'   in s)]

    iSplit0 = round(len(lTrain0)*rHoldout)
    iSplit1 = round(len(lTrain1)*rHoldout)

    lV0 = lTrain0[:iSplit0]
    lV1 = lTrain1[:iSplit1]
    lT0 = lTrain0[iSplit0:]
    lT1 = lTrain1[iSplit1:]

    return(lT0, lT1, lV0, lV1, lTest)

def LoadFiles(sSrc, lFiles):

    raaaData = None;

    iFiles = len(lFiles)
    for k in range(iFiles):
        sFile = lFiles[k]+'.pkl'
        (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sSrc,sFile),'rb') )
        if(raaaData==None):
            raaaData = numpy.empty((iFiles, raaData.shape[0], raaData.shape[1]))
        raaaData[k,:,:] =  raaData;

    return(raaaData)

def TrainClassifier(sShufflePath, sRatePath, iSensors, tlGeometry, oaLayersX, rHoldout=0.2, iBatches=100, iBatchFiles=1000, iBatchPatterns=100):

    rRate = 0.01
    rMomentum = 0.9
    rWeightDecay = 0.0001
    iFinalBatches = 2

    def CreateRandomModel(sModelName, iSensors, tlGeometry, rWeightInitScale = 0.001):

        # Create an empty list of layers
        oaLayers = []

        # First visible pattern size is number of electrodes
        iV = iSensors

        # For each layer in geometry...
        for (iDecimation, iH) in tlGeometry:

            # Create a random weight matrix
            raaW = numpy.random.randn(iDecimation*iV,iH)*rWeightInitScale

            iV = iH

            # Create a zeros bias vector
            raB  = numpy.random.randn(iH)*rWeightInitScale

            # Construct a layer
            oLayer = SequenceDecimatingNetwork.Layer(iDecimation, raaW, raB)

            # Add it to the list of layers
            oaLayers.append(oLayer) 

        # Create a sequence decimating network using this layer stack 
        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        return(oModel)

    # Get name for the model
    sModelName = GetModelName(sRatePath, iSensors, tlGeometry, "Layers")

    # If the pretrained model exists...
    if(os.path.exists(sModelName)):

        # Load layer stack
        f = open(sModelName,"rb")
        oaLayers = pickle.load(f)
        f.close()

        # Create a sequence decimating network using this layer stack 
        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        # Load model
        f = open(sModelName+"Sdn","rb")
        oSdn = pickle.load(f)
        f.close()

        print(oModel==oSdn)
        print(oModel.hash(), oSdn.hash())

    # Otherwise
    else:

        # Create a new model
        oModel = CreateRandomModel(sModelName, iSensors, tlGeometry, rWeightInitScale)

    iTrainIndex = 0;
    (lT0, lT1, lV0, lV1, lTest) = LoadTrainingShuffle(sShufflePath, rHoldout)

    iT0 = 0
    iT1 = 0

    # For each training batch...
    for iBatch in range(iBatches):

        lTrain = []
        raaaT = numpy.zeros((iBatchFiles,1,1))
        for k in range(iBatchFiles):
            if(k % 2):
                lTrain.append(lT0[iT0 % len(lT0)])
                iT0+=1
            else:
                lTrain.append(lT1[iT1 % len(lT1)])
                iT1+=1
            raaaT[k,0,0] = 'preictal' in lTrain[k]

        # Load training files
        raaaX = LoadFiles(sRatePath, lTrain)
        (iPatterns, iSamples, iFeatures) = raaaX.shape
        iD = 1
        for (iDecimation, iH) in tlGeometry:
            iD *= iDecimation

        # print("iDecimation={:d}".format(iD))

        raaaXt = numpy.zeros((iPatterns,iD,iFeatures))

        for p in range(iPatterns):
            iOffset = numpy.random.randint(iSamples-iD)
            raaaXt[p,:,:] = raaaX[p,iOffset:iOffset+iD,:]

        #print(oSdn.hash())
        #print(oModel.hash())

        # Run a training batch
        oSdn.Train(raaaXt, raaaT, iBatchPatterns, rRate, rMomentum, iBatch<iFinalBatches, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))
        #oModel.Train(raaaXt, raaaT, iBatchPatterns, rRate, rMomentum, iBatch<iFinalBatches, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))

    # Load train files
    lTrain = lT0+lT1
    raaaX = LoadFiles(sRatePath, lTrain)
    raaaXt = raaaX[:,:iD,:]    
    raY = numpy.squeeze(oModel.ComputeOutputs(raaaXt))

    f = open(os.path.join(sRatePath,'Train.csv'),'wt')

    for k in range(len(lTrain)):
        print("{:s}.mat,{:.6f}".format(lTrain[k],raY[k]),file=f)
    f.close()

    # Load test files
    raaaX = LoadFiles(sRatePath, lTest)
    raaaXt = raaaX[:,:iD,:]    
    raY = numpy.squeeze(oModel.ComputeOutputs(raaaXt))

    f = open(os.path.join(sRatePath,'Test.csv'),'wt')

    for k in range(len(lTest)):
        print("{:s}.mat,{:.6f}".format(lTest[k],raY[k]),file=f)
    f.close()

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

def PreprocessMatFiles(sSrc, sDst, rSampleFrequency=400, bDetrend=False):

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
        iSensors = oMat[lKeys[0]]['data'][0,0].shape[0]
        iSamples    = oMat[lKeys[0]]['data'][0,0].shape[1]
        rLength     = oMat[lKeys[0]]['data_length_sec'][0,0][0,0]   
        rFrequency  = oMat[lKeys[0]]['sampling_frequency'][0,0][0,0]

        return((raaData, iSensors, iSamples, rLength, rFrequency))

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
            (raaData, iSensors, iSamples, rLength, rFrequency) = LoadMat(sSrc + '\\' + f)

            # Compute the nearest integer decimation ratio
            iDecimationRatio = int(round(rFrequency/rSampleFrequency))

            # If detrending...
            if bDetrend:

                # Detrend along time axis
                raaData = scipy.signal.detrend(raaData, axis=1).astype(numpy.float32)

            # If decimating...
            if iDecimationRatio > 1:

                # Decimate using 6th order chebyshev IIR
                raaData = scipy.signal.decimate(raaData, iDecimationRatio, ftype='iir', n=6, axis=1).astype(numpy.float32)

            if(numpy.isnan(numpy.sum(raaData))):

                print("NAN detected")

            # If normalizing...
            if bNormalize:

                # For each electrode...
                for iSensor in range(iSensors):

                    # Force unit standard deviation
                    raaData[iSensor,:] /= raaData[iSensor,:].std()

                # Scale to specified peak to average ratio
                raaData = numpy.maximum(numpy.minimum(1,raaData/(rPeakAverageRatio)),-1).astype(numpy.float32)

                # Transform between zero and one
                raaData = (raaData+1)/2

            # Pickle a tuple with fields we want
            pickle.dump((raaData.T, rFrequency, sClass), open(sDstFile,'wb'))

        print('{:4d} of {:4d} {}'.format(iFile,len(lFiles),f))

Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(16,128),(2,128),(2,128),(2,128),(2,1)],.2,False)
#Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(16,128),(2,128),(2,128),(2,128),(2,1)],.2)
#Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(8192,1)],.2) #,(8,128),(8,1)],.2)
#Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(16,128),(8,1)], 0.2)#,(8,128),(8,1)],.2)
