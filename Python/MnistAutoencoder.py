# This class provides methods to train MNIST autoencoders, view their 
# operation, and inspect their first layer feature detectors.
#
# * MnistAutoencoder.RunExperiments() - Trains 16 autoencoders using a
#   variety of parameter settings.
# * MnistAutoencoder.View(sModel) - Displays input and output for the 
#   autoencoder filename specified by sModel.
# * MnistAutoencoder.ViewFeatureDetectors(sModel) - Displays the first
#   layer feature detectors learned by the model.

import pandas
import numpy
import RbmStack

# iEpochs = 100

# df = pandas.read_pickle("MNIST.pkl")

# raaX = numpy.array(df[0:784])

# # #raaX = numpy.random.randn(iSamples,iInputs)
# oaLayers = [RbmStack.Layer(raaX.shape[1],iEpochs),RbmStack.Layer(1000,iEpochs),RbmStack.Layer(30,iEpochs)]

# oOptions = RbmStack.Options(iEpochs)

# oRbmStack = RbmStack.RbmStack(oaLayers)
# oRbmStack.TrainAutoencoder(raaX, oOptions)

# #print(o.oaLayer[1].raaW)

class MnistAutoencoder:
        
    ## RunExperiments
    # Train a set of 16 MNIST models using various combinations of 
    # training epochs, stochastic sampling, dropout, and network topology.

    @staticmethod
    def RunExperiments():
        
        # Define a compact def to launch an experiment
        def Go(iEpochs, bSample, bDropout, iaSize):
            
            # Define Hinton MNIST dropout recipe
            raDropV = [       0.2,  0.0, 0.0,  0.0]
            raDropH = [       0.5,  0.5, 0.5,  0.5]

            # Launch experiment
            MnistAutoencoder.RbmExperiment(raaTrain, raaTest, iaSize, iEpochs, bSample, raDropV*bDropout, raDropH*bDropout)
        
        # Load the mnist data
        df = pandas.read_pickle("MNIST.pkl")

        raaTrain = numpy.array(df[df["subset"]==0].ix[:,0:784])
        raaTest  = numpy.array(df[df["subset"]==1].ix[:,0:784])

        # Measure the training data
        [iSamples, iFeatures] = raaTrain.shape
    
        # Launch lots of experiments
        # (iEpochs, bSample, bDropout, iaSize)
        Go( 10, 0, 0, [ iFeatures, 1000,  500,  250,   30])
        Go( 10, 0, 1, [ iFeatures, 1000,  500,  250,   30])
        Go( 10, 1, 0, [ iFeatures, 1000,  500,  250,   30])  
        Go( 10, 1, 1, [ iFeatures, 1000,  500,  250,   30])
        
        Go( 10, 0, 0, [ iFeatures, 2000, 1000,  500,   30])
        Go( 10, 0, 1, [ iFeatures, 2000, 1000,  500,   30])     
        Go( 10, 1, 0, [ iFeatures, 2000, 1000,  500,   30])
        Go( 10, 1, 1, [ iFeatures, 2000, 1000,  500,   30])
        
        Go( 50, 0, 0, [ iFeatures, 1000,  500,  250,   30])
        Go( 50, 0, 1, [ iFeatures, 1000,  500,  250,   30])
        Go( 50, 1, 0, [ iFeatures, 1000,  500,  250,   30])
        Go( 50, 1, 1, [ iFeatures, 1000,  500,  250,   30])
        
        Go( 50, 0, 0, [ iFeatures, 2000, 1000,  500,   30])
        Go( 50, 0, 1, [ iFeatures, 2000, 1000,  500,   30])
        Go( 50, 1, 0, [ iFeatures, 2000, 1000,  500,   30])
        Go( 50, 1, 1, [ iFeatures, 2000, 1000,  500,   30])

MnistAutoencoder.RunExperiments()

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
                                          
        # Create momentum schedule
        raMomentum = 0.9*numpy.ones((iEpochs,1)
        raMomentum[0:5] = 0.5
          
        # Set the number of training epochs
        oOptions.iEpochs = iEpochs
        
        # Set the event reporting callback
        oOptions.fEvent = @Report
        
        # Create a default training rate vector
        raRate = linspace(.1,.1,iEpochs)
        
        # Specify machine geometry
        oaLayer(1).sActivationUp = 'Logistic'
        oaLayer(1).sActivationDn = 'Logistic'
        oaLayer(1).iSize = iaSize(1)
        oOptions.oaLayer(1).raRate = raRate
        oOptions.oaLayer(1).raMomentum = raMomentum
        oOptions.oaLayer(1).raDropV = raDropV(1)*ones(iEpochs,1)
        oOptions.oaLayer(1).raDropH = raDropH(1)*ones(iEpochs,1)
        oOptions.oaLayer(1).baSample = bSample*ones(iEpochs,1)
        
        oaLayer(2).sActivationUp = 'Logistic'
        oaLayer(2).sActivationDn = 'Logistic'
        oaLayer(2).iSize = iaSize(2)
        oOptions.oaLayer(2).raRate = raRate
        oOptions.oaLayer(2).raMomentum = raMomentum
        oOptions.oaLayer(2).raDropV = raDropV(2)*ones(iEpochs,1)
        oOptions.oaLayer(2).raDropH = raDropH(2)*ones(iEpochs,1)
        oOptions.oaLayer(2).baSample = bSample*ones(iEpochs,1)
        
        oaLayer(3).sActivationUp = 'Logistic'
        oaLayer(3).sActivationDn = 'Logistic'
        oaLayer(3).iSize = iaSize(3)
        oOptions.oaLayer(3).raRate = raRate
        oOptions.oaLayer(3).raMomentum = raMomentum
        oOptions.oaLayer(3).raDropV = raDropV(3)*ones(iEpochs,1)
        oOptions.oaLayer(3).raDropH = raDropH(3)*ones(iEpochs,1)
        oOptions.oaLayer(3).baSample = bSample*ones(iEpochs,1)
        
        oaLayer(4).sActivationUp = 'Linear'
        oaLayer(4).sActivationDn = 'Logistic'
        oaLayer(4).iSize = iaSize(4)
        oOptions.oaLayer(4).raRate = raRate/100
        oOptions.oaLayer(4).raMomentum = raMomentum
        oOptions.oaLayer(4).raDropV = raDropV(4)*ones(iEpochs,1)
        oOptions.oaLayer(4).raDropH = raDropH(4)*ones(iEpochs,1)
        oOptions.oaLayer(4).baSample = bSample*ones(iEpochs,1) 
        
        oaLayer(5).iSize = iaSize(5)
        
        # Construct the object
        oModel = RbmStack(oaLayer)
        
        # Open the log file
        oLog = fopen(strcat(MnistAutoencoder.sModelPath, 'Log.txt'), 'at')
        
        # Infer dropout flag
        bDropout = max(raDropV+raDropH)>0
        
        # Build a filename for the model
        sName = sprintf('iEpochs=#d bSample=#d bDropout=#d (#d #d #d #d #d)', iEpochs, bSample, bDropout, iaSize)
        Log(sprintf('#s\n',sName))
        
        # Get clock time as a string
        sNow = datestr(now,'yyyy-mmm-dd HH MM SS')
        Log(sprintf('sNow= #s\n\n',sNow))
        
        # Summarize the machine geometry
        for iLayer = 1:length(oaLayer)-1
            Log(sprintf('iLayer=#d, iSizeV=#5d, iSizeH=#5d, sActivationUp=#10s, sActivationDn=#10s\n',...
                iLayer,...
                oaLayer(iLayer).iSize,...
                oaLayer(iLayer+1).iSize,...
                oaLayer(iLayer).sActivationUp,...
                oaLayer(iLayer).sActivationDn))
        
        # Newline
        Log(sprintf('\n'))
        
        # Train the object
        TrainAutoencoder(oModel, raaTrain, oOptions)
        
        Log(sprintf('\n'))
       
        # Save the trained autoencoder
        save(strcat(MnistAutoencoder.sModelPath,sName),'oModel')

        # Compute training and test set errors
        [rTrainRmse, rTrainError] = ComputeReconstructionError(oModel, raaTrain)
        [rTestRmse,  rTestError]  = ComputeReconstructionError(oModel, raaTest)

        # Report performance
        Log(sprintf('rTrainRmse= #.4f, rTrainError= #.4f\n', rTrainRmse, rTrainError))
        Log(sprintf('rTestRmse=  #.4f, rTestError=  #.4f\n', rTestRmse,  rTestError))
        Log(sprintf('\n'))

        # Summary string (used to assemble a table in word)
        Log(sprintf('XX: #d,#d,#d,#4d #4d #4d #4d #4d,', iEpochs, bSample, bDropout, iaSize))
        for iLayer=1:4
            Log(sprintf('#0.4f,', oModel.oaLayer(iLayer).raError(end)))
        end
        Log(sprintf('#0.4f,#0.4f,', rTrainRmse, rTrainError))
        Log(sprintf('#0.4f,#0.4f\n', rTestRmse,  rTestError))

        # Close log file
        fclose(oLog)
         
        # Log a line of output to both the console and the log file
        def Log(sLine):
                            
            fprintf(     '#s', sLine)
            fprintf(oLog,'#s', sLine)
        
        # Process a training event 
        def Report(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError):
            
            # Construct event report string
            Log(sprintf('iLayer=#d, iEpoch=#3d, bSample=#d, rDropV=#.2f, rDropH=#.2f, rRate=#.4f, rMomentum=#.4f, rRmse=#.4f, rError=#.4f\n', iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError))
    
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

