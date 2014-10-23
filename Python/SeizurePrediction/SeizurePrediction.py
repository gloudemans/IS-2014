import math
import numpy
import RbmStack

def Log(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError):

    print('iLayer={}, iEpoch={}, bSample={}, rDropV={}, rDropH={}, rRate={}, rMomentum={}, rRmse={}, rError={}'.format(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError))

# Define a function to exercise the class (crudely!)
def Tester(sFile, iSize):

    import pickle

    # Define classes used for RbmStack interfacing
    class Layer:

        def __init__(self, iSize, iEpochs, sActivationUp='Logistic'):
            self.iSize = iSize
            self.raaW = object
            self.sActivationUp = sActivationUp
            self.sActivationDn = "Logistic"
            self.raRmse     = numpy.zeros(iEpochs)
            self.raError    = numpy.zeros(iEpochs)

    class LayerOptions:

        def __init__(self, iEpochs):

            self.raDropV    = 0.0*numpy.ones(iEpochs)
            self.raDropH    = 0.0*numpy.ones(iEpochs)
            self.raMomentum = 0.9*numpy.ones(iEpochs)
            self.raMomentum[:5]=0.5;
            self.raRate     = 0.1*numpy.ones(iEpochs)
            self.baSample   = 0*numpy.ones(iEpochs)
            self.raRmse     = numpy.zeros(iEpochs)


    class Options:

        def __init__(self, iEpochs):

            self.iEpochs = iEpochs
            self.oaLayer = [LayerOptions(iEpochs), LayerOptions(iEpochs), LayerOptions(iEpochs)]  
            self.fEvent = Log

    # Specify epochs
    iEpochs = 50

    
    (raaTrain, iaTrain, raaTest, iaTest) = pickle.load( open( sFile, "rb" ) )
    print(raaTrain.shape)

    raaX = raaTrain

    # Create 784 x 1000 rbm layer
    oaLayers = [Layer(raaX.shape[1],iEpochs),Layer(1000,iEpochs), Layer(iSize,iEpochs)]

    # Create training options
    oOptions = Options(iEpochs)

    # Create RbmStack
    oRbmStack = RbmStack.RbmStack(oaLayers)

    # Train using the specified options
    oRbmStack.TrainAutoencoder(raaX, oOptions)

    # Compute training and test set errors
    (rTrainRmse, rTrainError) = oRbmStack.ComputeReconstructionError(raaTrain)
    (rTestRmse,  rTestError)  = oRbmStack.ComputeReconstructionError(raaTest)

    #     # Report performance
    oFile = open('Log.txt','at')
    print('sFile={}, iSize={:d}'.format(sFile,iSize),file=oFile)
    print('rTrainRmse= {:.4f}, rTrainError= {:.4f}'.format(rTrainRmse, rTrainError),file=oFile)
    print('rTestRmse=  {:.4f}, rTestError=  {:.4f}'.format(rTestRmse,  rTestError),file=oFile)
    oFile.close()

sFile = "Batch_0000.pkl" 
Tester(sFile,10)
Tester(sFile,20)
Tester(sFile,50)
Tester(sFile,100)
Tester(sFile,200)
Tester(sFile,500)
Tester(sFile,1000)
Tester(sFile,2000)


