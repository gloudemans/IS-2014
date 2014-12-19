import os
import random
import scipy.io     # for loadmat
import scipy.signal # for detrend
import numpy
import pickle
import RbmStack
import SequenceDecimatingNetwork

import matplotlib.pyplot as plt

# Performs all processing required to train a sequence decimating network for
# each patient, classify the test data, and form a consolidated file for
# upload to kaggle. 
#
# * sDatasetPath - path to root of dataset
# * rSampleFrequency - sample frequency target for preprocessing (Hz)
# * tlGeometry - tuple list defining network geometry for each layer (InputDecimation, OutputFeatures)
# * rHoldout - fraction of samples to hold out for validation
# * bRetrain - force pretraining even if prior result exists
#
# Arrange the .mat format data files distributed by Kaggle into the following
# directory structure.
#
# * sDatasetPath/Dog_1
# * sDatasetPath/Dog_2
# * sDatasetPath/Dog_3
# * sDatasetPath/Dog_4
# * sDatasetPath/Dog_5
# * sDatasetPath/Patient_1
# * sDatasetPath/Patient_2

def Go(sDatasetPath, rSampleFrequency, tlGeometry, rHoldout=0.2, iPretrainEpochs=50, iPretrainPatterns=20000, bRetrain=False):

    # List (patients directory name, sensor count) 
    lPatients = [('Dog_1',16),('Dog_2',16),('Dog_3',16),('Dog_4',16),('Dog_5',15),('Patient_1',15),('Patient_2',24)]

    # For each patient...
    for (sPatient,iSensors) in lPatients:

        # Path to patient data
        sPatientPath = os.path.join(sDatasetPath,sPatient)

        # Form path to sample rate directory
        sRatePath = os.path.join(sPatientPath, "{:d}Hz".format(round(rSampleFrequency)))

        # If the rate specific directory doesn't exist...
        if not os.path.isdir(sRatePath):

            # Make it
            os.makedirs(sRatePath)           

        # Preprocess files to desired sample rate if necesary 
        PreprocessMatFiles(sPatientPath, sRatePath, rSampleFrequency)

        # Get shuffled file list
        lShuffle = GetShuffle(sPatientPath)

        # Construct training, validation, test splits
        tSplits = SplitShuffle(lShuffle, rHoldout)

        # Pretrain an autoencoder with the specified geometry if necessary (layers saved to file)
        oaLayers = PretrainAutoencoder(sRatePath, tSplits, iSensors, tlGeometry, bRetrain, iPretrainEpochs, iPretrainPatterns)

        # Train classifier using the specified geometry (layers read from file)
        oModel = TrainClassifier(oaLayers, sRatePath, tSplits, iSensors, tlGeometry, bRetrain)

        # Process training, validation, and test files using the trained model
        ProcessAllFiles(oModel, sRatePath, tSplits, tlGeometry, sPatient)

    # Consolidate test results from all patients into one file
    ConsolidateTestResults(sDatasetPath, lPatients, rSampleFrequency)

# If sPatientPath/"Shuffle.csv" exists, rad it into a list and return,
# else create it and write a randomly shuffled list of all the .mat 
# file names in sPatientPath.
#
# * sPatientPath - specifies the path to patient data

def GetShuffle(sPatientPath):

    # Path to Shuffle.csv
    sShufflePath = os.path.join(sPatientPath, "Shuffle.csv")

    # If Shuffle.csv does not exist...
    if(os.path.exists(sShufflePath)):

        # Read file into list
        oFile = open(sShufflePath,'rt')
        lShuffle = oFile.read().splitlines();
        oFile.close()

    else:

        # Get names of all the .mat files
        lShuffle = [f[:-4] for f in os.listdir(sPatientPath) if(f.endswith('.mat'))]

        # Shuffle the file names
        random.shuffle(lShuffle)

        # Open Shuffle.csv
        oFile = open(sShufflePath,'wt')

        # For each filename...
        for s in lShuffle:

            oFile.write(s+"\n")

        # Close the file
        oFile.close()

    return(lShuffle)

# For interictal and preictal file types, extract the leading (1-rHoldout) 
# fraction of the files as the training subset and the trailing rHoldout 
# fraction of the files as the validation subset.
#
# * lFiles - specifies the shuffled filenames
# * rHoldout - specifies the validation holdout fraction

def SplitShuffle(lFiles, rHoldout):

    # Form separate lists of test, interictal, and preictal files
    lTest   = [s for s in lFiles if('test'       in s)]
    lTrain0 = [s for s in lFiles if('interictal' in s)]
    lTrain1 = [s for s in lFiles if('preictal'   in s)]

    # Number of files to retain in each training set
    iTrain0 = round(len(lTrain0)*(1-rHoldout))
    iTrain1 = round(len(lTrain1)*(1-rHoldout))

    # Partition the training and validation subsets
    lV0 = lTrain0[iTrain0:]
    lV1 = lTrain1[iTrain1:]
    lT0 = lTrain0[:iTrain0]
    lT1 = lTrain1[:iTrain1]

    # Return training interictal and preictal, validation interictal and preictal,
    # and test lists
    return(lT0, lT1, lV0, lV1, lTest)

# For each .mat file in the source directory, read the data from the .mat
# file, decimate it to the specified sample frequency, optionally detrend
# each sensor, normalize each sensor to have a fixed standard deviation, and clip
# at [-1,+1]. Store the preprocessed data, sample frequency, and class as a tuple 
# in a pickle file. 
#
# * sSrc - specifies the source directory for .mat files
# * sDst - specifies the destination directory for .pkl files
# * rSampleFrequency - specifies the target sample rate
# * bDetrend - specifies linear detrending if true

def PreprocessMatFiles(sSrc, sDst, rSampleFrequency, bDetrend=False):

    # Load the .mat file specified by sFile and return the
    # data array, the number of sensors, the number of samples,
    # the length, and the sampling frequency as a tuple.
    #
    # * sFile - specifies the file to load

    def LoadMat(sFile):

        # Load one file
        oMat = scipy.io.loadmat(sFile)

        # Find the variable name
        lKeys = [s for s in oMat.keys() if "segment" in s]

        raaData     = oMat[lKeys[0]]['data'][0,0]
        iSensors    = oMat[lKeys[0]]['data'][0,0].shape[0]
        iSamples    = oMat[lKeys[0]]['data'][0,0].shape[1]
        rLength     = oMat[lKeys[0]]['data_length_sec'][0,0][0,0]   
        rFrequency  = oMat[lKeys[0]]['sampling_frequency'][0,0][0,0]

        return((raaData, iSensors, iSamples, rLength, rFrequency))

    # Convert the specified file name into a string class target.
    #
    # * sFile - specifies the file     

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

            # If the filtered data contains not a number...
            if(numpy.isnan(numpy.sum(raaData))):

                # Report it
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

            # Pickle a tuple with the fields we want
            pickle.dump((raaData.T, rFrequency, sClass), open(sDstFile,'wb'))

            # Report preprocessing of this file
            print('Preprocessed {:4d} of {:4d} {}'.format(iFile,len(lFiles),f))

    # Report preprocessing completion
    print('Preprocessing of {:4d} files complete'.format(len(lFiles)))

# If the layers file exists and retraining is not specified, and read the layers from the file.
# Otherwise, pretrain a stack of layers as a sequence decimating network autoencoder and save the
# layers. Return the stack of layers.
#
# * sRatePath - specifies the path to the rate specific training data
# * tSplits - specifies a tuple with training, validation and test data splits
# * iSensors - specifies the number of sensors at the input layer
# * tlGeometry - specifies the layer geometry as a list of (decimation, feature) tuples
# * bRetrain - true to force retraining even if the layers file exists
# * iEpochs - specifies the number of epochs to train
# * iPatterns - specifies the number of patterns per epoch

def PretrainAutoencoder(sRatePath, tSplits, iSensors, tlGeometry, bRetrain, iEpochs, iPatterns):
    
    # Generate training patterns using the network specified by oalayers Otherwise, pretrain a stack of layers as a sequence decimating network autoencoder and save the
    # layers. Return the stack of layers.
    #
    # * sRatePath - path to the rate specific training data
    # * lT0 - list of interictal training files
    # * lT1 - list of preictal training files
    # * oaLayers - list of layers with which to process the input data to produce the training vector
    # * iPatterns - specifies the number of training patterns to generate
    # * iTotalDecimation - specifies the number of patterns per epoch
    # * iSensors - specifies the number of sensors at the input layer

    def GenerateTrainingPatterns(sRatePath, lT0, lT1, oaLayers, iPatterns, iTotalDecimation, iSensors):

        # Numbers of interical and preictal training patterns to generate
        iT0 = round(iPatterns/2)
        iT1 = iPatterns - iT0

        # Create an array to hold the output patterns
        raaaX = numpy.zeros((iPatterns, iTotalDecimation, iSensors))

        # Get total file count
        iFiles = len(lT0)+len(lT1)

        # Create a permutation index
        ia = numpy.random.permutation(iPatterns)

        # Clear the pattern count
        iPattern  = 0

        # For each interictal file...
        for k in range(len(lT0)):

            # Get filename
            sFile = lT0[k]+'.pkl'

            # Load the data
            (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sRatePath,sFile),'rb') )

            # Measure the data
            (iSamples, iSensors) = raaData.shape

            # While more samples are required from this file...
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

            # While more samples are required from this file...
            while(iPattern<(k+1)*iT1/len(lT1)+iT0):

                # Compute a random offset
                iOffset = random.randrange(iSamples-iTotalDecimation)

                # Save this pattern
                raaaX[ia[iPattern],:,:] = raaData[iOffset:iOffset+iTotalDecimation,:]

                # Next pattern
                iPattern += 1

        # Construct a sequence decimating network from the layer stack
        oSdn = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        # Apply the sequence decimating network to the training inputs to create training sample inputs
        # for the current layer
        raaaY = oSdn.ComputeOutputs(raaaX)

        # Reconstruct the input data from the network output
        raaaXr = oSdn.ComputeInputs(raaaY)

        # Compute the root mean squared reconstruction error
        rRmse = numpy.sqrt(numpy.mean((raaaX[:]-raaaXr[:])**2))

        # Report the reconstruction error of the network with this training batch
        print("Reconstruction RMSE = {:.6f}".format(rRmse))

        # Measure the network output
        (iP,iSamples,iSensors) = raaaY.shape

        # Reshape the network output to 2D
        raaY = raaaY.reshape(iP, iSamples*iSensors)

        return(raaY)

    # Construct layer file name
    sLayerFile = sRatePath + "\\Layers.pkl"

    # If the layer file already exists...
    if(bRetrain or not os.path.exists(sLayerFile)):

        # Create training options
        oOptions = RbmStack.Options(iEpochs)

        # Load training splits
        (lT0, lT1, lV0, lV1, lTest) = tSplits

        # Initialize total decimation ratio
        iTotalDecimation = 1

        # Create a list to store sequence decimation layers
        oaLayers = []; 

        # Initial visible layer size is sensor count
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
            raaX = GenerateTrainingPatterns(sRatePath, lT0, lT1, oaLayers, iPatterns, iTotalDecimation, iSensors)

            # Compute the standard deviation of training patterns
            rStd = numpy.std(raaX[:])

            print("\nPretraining Layer {:d} with standard deviation {:.4f}\n".format(iLayer, rStd))

            # Train the autoencoder
            oRbm.TrainAutoencoder(raaX, oOptions)

            # Add the new weights to the layer stack
            oaLayers.append(SequenceDecimatingNetwork.Layer(iDecimation, oRbm.oaLayer[0].raaW, oRbm.oaLayer[0].raH, oRbm.oaLayer[0].raV))

            # Hidden outputs are next layer inputs
            iV = iH

        # Generate training patterns once more to measure the error performance
        raaX = GenerateTrainingPatterns(sRatePath, lT0, lT1, oaLayers, iPatterns, iTotalDecimation, iSensors)

        # Save layers
        f = open(sLayerFile,"wb")
        pickle.dump(oaLayers, f)
        f.close()

    else:

        # Load layers
        f = open(sLayerFile,"rb")
        oaLayers = pickle.load(f)
        f.close()

    return(oaLayers)

# Perform supervised training of the sequence decimating network using backpropagation with
#  stochastic gradient descent.
#
# * oaLayers - specifies the layer stack to use for the sequence decimating network parameters
# * sRatePath - path to the rate specific training data
# * tSplits - training, validation and test splits
# * iSensors - number of sensors at the input layer
# * tlGeometry - list of (decimation, features) tuples to define the network geometry
# * iBatches - number of batches to use for training
# * iBatchFiles - maximum number of files to load during training (used to cope with inadequate memory)
# * iBatchPatterns - number of patterns in a batch

def TrainClassifier(oaLayers, sRatePath, tSplits, iSensors, tlGeometry, bRetrain=False, bRandomModel=False, iBatches=100, iBatchFiles=1000, iBatchPatterns=10000):

    # Learning rate
    rRate = 0.001

    # Learning momentum
    rMomentum = 0.9

    # Learning weight decay
    rWeightDecay = 0.0001

    # Number of training batches during which to train only the final layer of the network
    iFinalBatches = 2

    # Create a sequence decimating network with the specified geometry
    # and small random parameter.
    #
    # * iSensors - specifies the number of sensors at the input
    # * tlGeometry - list of (decimation, features) tuples specifying the network geometry
    # * rWeightInitScale - standard deviation to use for the parameter values

    def CreateRandomModel(iSensors, tlGeometry, rWeightInitScale = 0.01):

        # Create an empty list of layers
        oaLayers = []

        # First visible pattern size is number of electrodes
        iV = iSensors

        # For each layer in geometry...
        for (iDecimation, iH) in tlGeometry:

            # Create a random weight matrix
            raaW = numpy.random.randn(iDecimation*iV,iH)*rWeightInitScale

            # Hidden layer becomes the next visble layer
            iV = iH

            # Create a zeros bias vector
            raH  = numpy.random.randn(iH)*rWeightInitScale

            # Construct a layer
            oLayer = SequenceDecimatingNetwork.Layer(iDecimation, raaW, raH)

            # Add it to the list of layers
            oaLayers.append(oLayer) 

        # Create a sequence decimating network using this layer stack 
        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        return(oModel)

    # Construct layer file name
    sModelFile = sRatePath + "\\Model.pkl"

    # If the layer file already exists...
    if(bRetrain or not os.path.exists(sModelFile)):

        # If the pretrained model exists...
        if(bRandomModel):

            # Create a new model
            oModel = CreateRandomModel(iSensors, tlGeometry)

        # Otherwise
        else:   

            # Create a sequence decimating network using this layer stack 
            oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

        # Get the file splits
        (lT0, lT1, lV0, lV1, lTest) = tSplits

        # Clear the traing file indices
        iT0 = 0
        iT1 = 0

        # For each training batch...
        for iBatch in range(iBatches):

            # Clear the list of training files
            lTrain = []

            # Array of class targets
            raaaT = numpy.zeros((iBatchFiles,1,1))

            # For each batch file...
            for k in range(iBatchFiles):

                # If even...
                if(k % 2):

                    # Append an interictal file to the list
                    lTrain.append(lT0[iT0 % len(lT0)])
                    iT0+=1
                else:
                    # Append a preictal file tio the list
                    lTrain.append(lT1[iT1 % len(lT1)])
                    iT1+=1

                # Add the corrsponding class target
                raaaT[k,0,0] = 'preictal' in lTrain[k]

            # Load training files
            raaaX = LoadFiles(sRatePath, lTrain)

            # Measure the training files
            (iPatterns, iSamples, iFeatures) = raaaX.shape

            # Compute the overall decimation ratio
            iD = 1
            for (iDecimation, iH) in tlGeometry:
                iD *= iDecimation

            # Report the decimation ratio
            print("iDecimation={:d}".format(iD))

            # Create an array for training targets
            raaaXt = numpy.zeros((iPatterns,iD,iFeatures))

            # For each required pattern...
            for p in range(iPatterns):

                # Compute an offset from which to extract the pattern
                iOffset = numpy.random.randint(iSamples-iD)

                # Add this pattern to the training targets
                raaaXt[p,:,:] = raaaX[p,iOffset:iOffset+iD,:]

            # Run a training batch
            oModel.Train(raaaXt, raaaT, iBatchPatterns, rRate, rMomentum, iBatch<iFinalBatches, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))

        # Save layers
        f = open(sModelFile,"wb")
        pickle.dump(oModel.oaLayers, f)
        f.close()

    else:

        # Load layers
        f = open(sModelFile,"rb")
        oaLayers = pickle.load(f)
        f.close()

        # Create a sequence decimating network using this layer stack 
        oModel = SequenceDecimatingNetwork.SequenceDecimatingNetwork(oaLayers)

    # Return the model
    return(oModel)

    # Construct sequence decimating network from layers
    oModel = SequenceDecimatingNetwork(oaLayers)

# Load the specified list of data files from the specified directory and return a numpy
# array with the data.
#
# * sSrc - specifies the directory from which to load files
# * lFiles - specifies the list of files to load

def LoadFiles(sSrc, lFiles):

    # Null the data array
    raaaData = None

    # Measure the file list 
    iFiles = len(lFiles)

    # For each file in the list...
    for k in range(iFiles):

        # Make the filename
        sFile = lFiles[k]+'.pkl'

        # Load the two dimensional data for a file
        (raaData, rFrequency, sClass) = pickle.load( open(os.path.join(sSrc,sFile),'rb') )

        # If the data array is null...
        if(raaaData==None):

            # Create it with the correct shape
            raaaData = numpy.empty((iFiles, raaData.shape[0], raaData.shape[1]))

        # Insert the two dimensional array as one pattern in the three dimensional array
        raaaData[k,:,:] = raaData;

    # return the three dimensional array
    return(raaaData)

# Compute points on ROC curve.
# Just added this. It doesn't seem to be working.

def ComputeRocCurve(sRatePath, sFile):
    
    # Key function to enable sorting on the prediction field of the list
    def getkey(item):
        return(item[1])

    # Open the file
    f = open(os.path.join(sRatePath,sFile),'rt')
    l = f.read().splitlines()
    f.close()

    # Form a list of [filename, prediction, actual]
    l = [[s.split(',')[0], float(s.split(',')[1]), int('preictal' in s.lower())] for s in l]

    # Sort on the prediction field
    l.sort(key=getkey)

    raX = numpy.zeros(len(l))
    raY = numpy.zeros(len(l))

    # Clear positive and negative counters
    iPositives = 0
    iNegatives = 0

    # For each threshold...
    for k in range(0,len(l)-1):

        # At threshold k, the classifier indicates k negatives and len-k positives
        iPositives +=   l[k][2]
        iNegatives += 1-l[k][2]

    # For each threshold...
    for k in range(0,len(l)-1):

        # Clear false positives counter
        iFalsePositives = 0
        iTruePositives  = 0

        # For each above threshold value...
        for j in range(k, len(l)):

            # True positive if j>=k and true result was positive
            iTruePositives  +=   l[j][2]            
            iFalsePositives += 1-l[j][2]

        raX[k] = iFalsePositives/iNegatives
        raY[k] = iTruePositives /iPositives



    # Return points on the ROC curve
    return((raX, raY))

def ProcessAllFiles(oModel, sRatePath, tSplits, tlGeometry, sPatient, bPlot=False):

    # Compute the overall decimation ratio
    iD = 1
    for (iDecimation, iH) in tlGeometry:
        iD *= iDecimation

    # Get file splits
    (lT0, lT1, lV0, lV1, lTest) = tSplits

    # Process all training files with model
    ProcessFiles(oModel, sRatePath, lT0+lT1, "Train.csv", iD)

    # Process all training files with model
    ProcessFiles(oModel, sRatePath, lV0+lV1, "Validation.csv", iD) 

    # Process all test files with model
    ProcessFiles(oModel, sRatePath, lTest, "Test.csv", iD)           

    # Compute training ROC curve
    (raTX, raTY) = ComputeRocCurve(sRatePath, "Train.csv")

    # Compute validation ROC curve
    (raVX, raVY) = ComputeRocCurve(sRatePath, "Validation.csv")

    if(bPlot):

        # Plot ROC curves
        plt.plot(raTX, raTY,  label="Training") 
        plt.plot(raVX, raVY,  label="Validation")
        plt.plot([0,1],[0,1], label="Reference")
        plt.title("ROC Curve for {:s}".format(sPatient))
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.legend(loc=4)
        plt.grid()
        plt.show()



def ProcessFiles(oModel, sRatePath, lFiles, sFile, iDecimation):

    # Load the files
    raaaX = LoadFiles(sRatePath, lFiles)

    # Extract test patterns from the initial samples of each pattern
    raaaXt = raaaX[:,:iDecimation,:]    

    # Compute sequence decimating network outputs
    raY = numpy.squeeze(oModel.ComputeOutputs(raaaXt))

    # Open the output file
    f = open(os.path.join(sRatePath,sFile),'wt')

    # For each file...
    for k in range(len(lFiles)):

        # Save comma separated filename, prediction pairs
        print("{:s}.mat,{:.6f}".format(lFiles[k], raY[k]), file=f)
    
    # Close the file
    f.close()

# Consolidate test results from all patients into a single file.
#
# Reads patient test result from:
#  sDatasetPath/sPatient/sRate/"Test.csv"
#
# Writes consolidated results to:
#  sDatasetPath/"Upload.csv"

def ConsolidateTestResults(sDatasetPath, lPatients, rSampleFrequency):

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

    # Close the file
    fOut.close()

Go('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Raw',20,[(16,128),(2,128),(2,128),(2,128),(2,1)])   
