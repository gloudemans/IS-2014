# This class provides methods to train MNIST autoencoders, view their 
# operation, and inspect their first layer feature detectors.
#
# * MnistAutoencoder.RunExperiments() - Trains 16 autoencoders using a
#   variety of parameter settings.
# * MnistAutoencoder.View(sModel) - Displays input and output for the 
#   autoencoder filename specified by sModel.
# * MnistAutoencoder.ViewFeatureDetectors(sModel) - Displays the first
#   layer feature detectors learned by the model.

import datetime
import pandas
import numpy
import RbmStack

class MnistAutoencoder:
        
    ## RunExperiments
    # Train a set of 16 MNIST models using various combinations of 
    # training epochs, stochastic sampling, dropout, and network topology.

    @staticmethod
    def RunExperiments():
        
        # Define a compact def to launch an experiment
        def Go(iEpochs, bSample, bDropout, iaSize):
            
            # Define Hinton MNIST dropout recipe
            raDropV = numpy.asarray([0.2,  0.0, 0.0,  0.0])
            raDropH = numpy.asarray([0.5,  0.5, 0.5,  0.5])

            # Launch experiment
            MnistAutoencoder.RbmExperiment(raaTrain, raaTest, iaSize, iEpochs, bSample, raDropV*bDropout, raDropH*bDropout)
        
        print('Load Starting...')

        # Load the mnist data
        df = pandas.read_pickle("../Datasets/MNIST/MNIST.pkl")

        # Randomly permute the rows
        df.reindex(numpy.random.permutation(df.index))

        raaTrain = numpy.array(df[df["subset"]==0].ix[:,0:784])/256.0
        raaTest  = numpy.array(df[df["subset"]==1].ix[:,0:784])/256.0

        print('Load Complete...')

        # Measure the training data
        [iSamples, iFeatures] = raaTrain.shape
    
        # Launch lots of experiments
        # (iEpochs, bSample, bDropout, iaSize)
        Go( 10, 0, 0, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        Go( 10, 0, 1, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        Go( 10, 1, 0, numpy.array([ iFeatures, 1000,  500,  250,   30])) 
        Go( 10, 1, 1, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        
        Go( 10, 0, 0, numpy.array([ iFeatures, 2000, 1000,  500,   30]))
        Go( 10, 0, 1, numpy.array([ iFeatures, 2000, 1000,  500,   30]))     
        Go( 10, 1, 0, numpy.array([ iFeatures, 2000, 1000,  500,   30]))
        Go( 10, 1, 1, numpy.array([ iFeatures, 2000, 1000,  500,   30]))
        
        Go( 50, 0, 0, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        Go( 50, 0, 1, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        Go( 50, 1, 0, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        Go( 50, 1, 1, numpy.array([ iFeatures, 1000,  500,  250,   30]))
        
        Go( 50, 0, 0, numpy.array([ iFeatures, 2000, 1000,  500,   30]))
        Go( 50, 0, 1, numpy.array([ iFeatures, 2000, 1000,  500,   30]))
        Go( 50, 1, 0, numpy.array([ iFeatures, 2000, 1000,  500,   30]))
        Go( 50, 1, 1, numpy.array([ iFeatures, 2000, 1000,  500,   30]))

    ## RbmExperiment
    # Train an MNIST model with the specified training options.
    #
    # * raaTrain - specifies the training data
    # * raaTest - specifies the test data
    # * iaSize - specifies the layers sizes
    # * iEpochs - specifies the number of training epochs
    # * bSample - specifies stochastic sampling
    # * raDropV - specifies the visible layer dropout probabilities
    # * raDropH - specifies the hidden layer dropout probabilities
    
    def RbmExperiment(raaTrain, raaTest, iaSize, iEpochs, bSample, raDropV, raDropH):

        # Open the log file
        oLog = open('Log.txt', 'at')

        # Define classes used for RbmStack interfacing
        class Layer:

            def __init__(self, iSize, sActivationUp='Logistic', sActivationDn='Logistic'):
                self.iSize = iSize
                self.raaW = object
                self.sActivationUp = sActivationUp
                self.sActivationDn = sActivationDn

        class LayerOptions:

            def __init__(self, raDropV, raDropH, raMomentum, raRate, baSample):

                self.raDropV    = raDropV
                self.raDropH    = raDropH
                self.raMomentum = raMomentum
                self.raRate     = raRate
                self.baSample   = baSample

        class Options:

            def __init__(self, iEpochs, oaLayer, fEvent=None):

                self.iEpochs = iEpochs
                self.oaLayer = oaLayer
                self.fEvent  = fEvent

        # Log a line of output to both the console and the log file
        def Log(sLine):
                            
            print('{:s}'.format(sLine),end="")
            print('{:s}'.format(sLine),end="", file=oLog)

        # Process a training event 
        def Report(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError):

            # Construct event report string
            Log('iLayer={:d}, iEpoch={:3d}, bSample={:d}, rDropV={:.2f}, rDropH={:.2f}, rRate={:.4f}, rMomentum={:.4f}, rRmse={:.4f}, rError={:.4f}\n'.format(
                iLayer, iEpoch, int(bSample), rDropV, rDropH, rRate, rMomentum, rRmse, rError))   

        # Create momentum schedule
        raMomentum = 0.9*numpy.ones(iEpochs)
        raMomentum[0:5] = 0.5
                 
        # Create a default training rate vector
        raRate = numpy.linspace(.1,.1,iEpochs)

    #     # Specify machine geometry
    #     oaLayer(1).sActivationUp = 'Logistic'
    #     oaLayer(1).sActivationDn = 'Logistic'
    #     oaLayer(1).iSize = iaSize(1)
    #     oOptions.oaLayer(1).raRate = raRate
    #     oOptions.oaLayer(1).raMomentum = raMomentum
    #     oOptions.oaLayer(1).raDropV = raDropV(1)*ones(iEpochs,1)
    #     oOptions.oaLayer(1).raDropH = raDropH(1)*ones(iEpochs,1)
    #     oOptions.oaLayer(1).baSample = bSample*ones(iEpochs,1)

    #     oaLayer(2).sActivationUp = 'Logistic'
    #     oaLayer(2).sActivationDn = 'Logistic'
    #     oaLayer(2).iSize = iaSize(2)
    #     oOptions.oaLayer(2).raRate = raRate
    #     oOptions.oaLayer(2).raMomentum = raMomentum
    #     oOptions.oaLayer(2).raDropV = raDropV(2)*ones(iEpochs,1)
    #     oOptions.oaLayer(2).raDropH = raDropH(2)*ones(iEpochs,1)
    #     oOptions.oaLayer(2).baSample = bSample*ones(iEpochs,1)
        
    #     oaLayer(3).sActivationUp = 'Logistic'
    #     oaLayer(3).sActivationDn = 'Logistic'
    #     oaLayer(3).iSize = iaSize(3)
    #     oOptions.oaLayer(3).raRate = raRate
    #     oOptions.oaLayer(3).raMomentum = raMomentum
    #     oOptions.oaLayer(3).raDropV = raDropV(3)*ones(iEpochs,1)
    #     oOptions.oaLayer(3).raDropH = raDropH(3)*ones(iEpochs,1)
    #     oOptions.oaLayer(3).baSample = bSample*ones(iEpochs,1)
        
    #     oaLayer(4).sActivationUp = 'Linear'
    #     oaLayer(4).sActivationDn = 'Logistic'
    #     oaLayer(4).iSize = iaSize(4)
    #     oOptions.oaLayer(4).raRate = raRate/100
    #     oOptions.oaLayer(4).raMomentum = raMomentum
    #     oOptions.oaLayer(4).raDropV = raDropV(4)*ones(iEpochs,1)
    #     oOptions.oaLayer(4).raDropH = raDropH(4)*ones(iEpochs,1)
    #     oOptions.oaLayer(4).baSample = bSample*ones(iEpochs,1) 
        
    #     oaLayer(5).iSize = iaSize(5)

        oaLayer = \
            [
                Layer(iaSize[0]),
                Layer(iaSize[1]),
                Layer(iaSize[2]),
                Layer(iaSize[3],'Linear','Logistic'),
                Layer(iaSize[4])
            ]

        oOptions = Options(
            iEpochs,
            [ 
                LayerOptions(raDropV[0]*numpy.ones((iEpochs)), raDropH[0]*numpy.ones((iEpochs)), raMomentum, raRate, bSample*numpy.ones((iEpochs))),
                LayerOptions(raDropV[1]*numpy.ones((iEpochs)), raDropH[1]*numpy.ones((iEpochs)), raMomentum, raRate, bSample*numpy.ones((iEpochs))),
                LayerOptions(raDropV[2]*numpy.ones((iEpochs)), raDropH[2]*numpy.ones((iEpochs)), raMomentum, raRate, bSample*numpy.ones((iEpochs))),
                LayerOptions(raDropV[3]*numpy.ones((iEpochs)), raDropH[3]*numpy.ones((iEpochs)), raMomentum, raRate/100, bSample*numpy.ones((iEpochs))) 
            ],
            Report
            ) 
        
        # Construct the object
        oModel = RbmStack.RbmStack(oaLayer)

        # Infer dropout flag
        bDropout = max(raDropV+raDropH)>0
        
        # Build a filename for the model
        sName = 'iEpochs={:d} bSample={:d} bDropout={:d} ({:d} {:d} {:d} {:d} {:d})\n'.format(iEpochs, bSample, bDropout, iaSize[0], iaSize[1], iaSize[2], iaSize[3], iaSize[4])
        Log(sName)
        
        # Get clock time as a string
        sNow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Log('{0}\n'.format(sNow))

        # Summarize the machine geometry
        for iLayer in range(0,len(oaLayer)-1):
            Log('iLayer={:d}, iSizeV={:5d}, iSizeH={:5d}, sActivationUp={:10s}, sActivationDn={:10s}\n'.format(
                iLayer,
                oaLayer[iLayer].iSize,
                oaLayer[iLayer+1].iSize,
                oaLayer[iLayer].sActivationUp,
                oaLayer[iLayer].sActivationDn))
             
        # Train the object
        oModel.TrainAutoencoder(raaTrain[:,:], oOptions)
        
        Log('\n'.format())
       
    #     # Save the trained autoencoder
    #     save(strcat(MnistAutoencoder.sModelPath,sName),'oModel')

        # Compute training and test set errors
        (rTrainRmse, rTrainError) = oModel.ComputeReconstructionError(raaTrain[:,:])
        (rTestRmse,  rTestError)  = oModel.ComputeReconstructionError(raaTest)

    #     # Report performance
        Log('rTrainRmse= {:.4f}, rTrainError= {:.4f}\n'.format(rTrainRmse, rTrainError))
        Log('rTestRmse=  {:.4f}, rTestError=  {:.4f}\n'.format(rTestRmse,  rTestError))
        Log('')

        # Summary string (used to assemble a table in word)
        Log('XX: {:d},{:d},{:d},{:4d} {:4d} {:4d} {:4d} {:4d},'.format(iEpochs, bSample, bDropout, iaSize[0], iaSize[1], iaSize[2], iaSize[3], iaSize[4]))
        # for iLayer in range(4)
        #     Log(sprintf('{:0.4f,}', oModel.oaLayer[iLayer].raError[end]))

        Log('{:0.4f},{:0.4f},{:0.4f},{:0.4f}\n'.format(rTrainRmse, rTrainError,rTestRmse,  rTestError))

        # Close log file
        oLog.close()
    
MnistAutoencoder.RunExperiments()

    # ## View
    # # Load the mnist test data and the specified model, autoencode the
    # # test data and display the resulting digits side by side for comparison.
    # #
    # # * sFile - specifies the file from which to load the model
    
    # def View(sModel)
        
    #     # Load the mnist data
    #     load('Mnist')
        
    #     # Load the trained model
    #     load(strcat(MnistAutoencoder.sModelPath,sModel))
        
    #     # Autoencode the test digits
    #     raaP = Autoencode(oModel, raaTest)
        
    #     # Prompt 
    #     fprintf('Press spacebar to advance, Ctrl-C to quit.\n')
        
    #     # Make a new figure
    #     figure
                    
    #     # For each test sample...
    #     for iSample = 1:size(raaP,1)
            
    #         # Form test digit into image
    #         raaL = reshape(raaTest(iSample,:),28,28)'
            
    #         # Form reconstruction into image
    #         raaR = reshape(raaP(    iSample,:),28,28)'   
            
    #         # Plot original
    #         subplot(1,2,1)
    #         imagesc(raaL,[0 1]) colormap gray axis equal axis off 
    #         title('Original')
            
    #         # Plot reconstruction
    #         subplot(1,2,2)
    #         imagesc(raaR,[0 1]) colormap gray axis equal axis off
    #         title('Reconstruction')

    #         # Force the plot to draw
    #         drawnow
            
    #         # Wait for keypress
    #         pause
    
    # ## ViewFeatureDetectors
    # # Load the specified model and plot tiles of its input layer feature detectors.
    # #
    # # * sFile - specifies the file from which to load the model
    
    # def ViewFeatureDetectors(sModel)
        
    #     # Load the trained model
    #     load(strcat(MnistAutoencoder.sModelPath,sModel))
        
    #     # Get the first layer weight vectors
    #     raaW = oModel.oaLayer(1).raaW
        
    #     # Count the feature detectors
    #     iDetectors = size(raaW,2)
        
    #     raaX = ones(30*25,30*40)*0.5
        
    #     # For each detector...
    #     for iDetector = 1:iDetectors-1
            
    #         # Form into an image
    #         raaF = reshape(raaW(1:end-1,iDetector),28,28)'
                      
    #         iX = mod(iDetector-1,25)
    #         iY = floor((iDetector-1)/25)
            
    #         raaX(iX*30+(1:28),iY*30+(1:28)) = raaF

            
    #     figure, imagesc(raaX,[0 1]) colormap gray axis equal axis off

