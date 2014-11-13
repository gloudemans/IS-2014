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
    lPatients = [('Dog_1',16),('Dog_2',16),('Dog_3',16),('Dog_4',16),('Dog_5',15),('Patient_1',15),('Patient_2',24)]

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

def TrainClassifier(sShufflePath, sRatePath, iElectrodes, tlGeometry, rHoldout=0.2, iBatches=10, iBatchFiles=1000, iBatchPatterns=1000):

    rWeightInitScale = 0.01
    rRate = 0.01
    rMomentum = 0.5

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

    def CreateModel(sModelName, iElectrodes, tlGeometry, rWeightInitScale):

        # Create an empty list of layers
        oaLayers = []

        # First visible pattern size is number of electrodes
        iV = iElectrodes

        # For each layer in geometry...
        for (iDecimation, iH) in tlGeometry:

            # Create a random weight matrix
            raaW = numpy.random.randn(iDecimation*iV,iH)*rWeightInitScale

            iV = iH

            # Create a zeros bias vector
            raB  = numpy.random.randn(iH)*rWeightInitScale

            # Construct a layer
            oLayer = SequenceDecimatingNetwork.SequenceDecimatingNetwork.Layer(iDecimation, raaW, raB)

            # Add it to the list of layers
            oaLayers.append(oLayer) 

        # Create a sequence decimating network using this layer stack 
        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        return(oModel)

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

    # Get name for the model
    sModelName = GetModelName(sRatePath, iElectrodes, tlGeometry)

    # If the model already exists...
    if(os.path.exists(sModelName)):

        # Load it
        f = open(sModelName,"rb")
        oModel = pickle.load(f)
        f.close()

    # Otherwise
    else:

        # Create a new model
        oModel = CreateModel(sModelName, iElectrodes, tlGeometry, rWeightInitScale)

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
                raaaT[k,0,0] = 0
                iT0+=1
            else:
                lTrain.append(lT1[iT1 % len(lT1)])
                iT1+=1
                raaaT[k,0,0] = 1
            #raaaT[k,0,0] = 'preictal' in lTrain[k]

        # Load training files
        raaaX = LoadFiles(sRatePath, lTrain)
        (iPatterns, iSamples, iFeatures) = raaaX.shape
        iD = 8192
        raaaXt = numpy.zeros((iPatterns,iD,iFeatures))

        for p in range(iPatterns):
            iOffset = 0 #numpy.random.randint(iSamples-iD)
            raaaXt[p,:,:] = raaaX[p,iOffset:iOffset+iD,:]

        rMomentum = .9

        #raaaXt = raaaX[:,:256,:]

        # Run a training batch
        oModel.Train(raaaXt, raaaT, iBatchPatterns, rRate, rMomentum, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))

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
#Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(8192,1)],.2) #,(8,128),(8,1)],.2)
