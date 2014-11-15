import math
import numpy
try:
    import cudamat
    cudamat.init()
    cudamat.CUDAMatrix.init_random(seed = 42)
    bCudamatLoaded = True
except ImportError:
    bCudamatLoaded = False

#RbmStack
# This class implements a stack of restricted Boltzmann machines. It 
# provides methods to train the stack using a greedy layerwise
# approach, and to perform inference using the stack as an autoencoder.
#
# The RbmStack provides the following features:
#
# * Allows specification of architectures having varying numbers of
#   layers with specified size and activation functions.
# * Optionally implements dropout of visible and/or hidden neurons
#   during training.
# * Callback mechanism to specify the learning rate, momentum, dropout 
#   probability, and hidden layer sampling technique for each layer.
# * Callback mechanism for reporting training progress
#   and performance metrics.

# Define classes used for RbmStack interfacing
class Layer:

    def __init__(self, iV, iH, rInitialWeightVariance=0.1, sActivationUp='Logistic', sActivationDn='Logistic'):

        # Set visible and hidden layer sizes
        self.iV = iV
        self.iH = iH

        # Initialize random weights
        self.raaW = numpy.random.randn(iV,iH) * rInitialWeightVariance

        # Clear the biases
        self.raV = numpy.zeros(iV)
        self.raH = numpy.zeros(iH)

        # Set activation functions
        self.sActivationUp = sActivationUp
        self.sActivationDn = sActivationDn

class Options:

    # Default training parameters callback function
    def fTrainingParameters(iLayer, iEpoch):

        # Default learning rate
        rRate = 0.1

        # If first five epochs...
        if(iEpoch<5):

            # Use less momentum
            rMomentum = 0.5
        else:
            # Use more momentum
            rMomentum = 0.9

        # No visible layer dropout
        rDropV  = 0

        # No hidden layer dropout
        rDropH  = 0

        # No stochastic sampling of the hidden layer
        bSample = 1

        # Return training parameters
        return(rRate, rMomentum, rDropV, rDropH, bSample)

    # Default epoch reporting callback function
    def fEpochReport(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse):

        # Print a summary of training performence
        print('iLayer={}, iEpoch={}, bSample={}, rDropV={}, rDropH={}, rRate={}, rMomentum={}, rRmse={}'.format(
            iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse))

    def __init__(self, iEpochs, fTrainingParameters=fTrainingParameters, fEpochReport=fEpochReport):

        self.iEpochs = iEpochs
        self.fTrainingParameters = fTrainingParameters
        self.fEpochReport = fEpochReport
            
class RbmStack:
  
    ## RbmStack
    # Construct a new object with the specified configuration.
    #
    # * oaLayer - specifies the network configuration.
    
    def __init__(self, oaLayer, bUseGpu=False):

        # Layer properties
        self.oaLayer = oaLayer

        # Set the GPU flag
        self.bUseGpu = bUseGpu and bCudamatLoaded

        # Number of samples per training batch
        self.iBatchSamples = 100
        
        # Normalize the dropout gradient by the number of weight updates
        # rather than the number of training samples.
        self.bNormalizeDropoutGradient = False
                          
    ## TrainAutoencoder
    # Perform greedy pre-training of the network using the specified
    # training data and options.

    def TrainAutoencoder(self, raaX, oOptions):

        # If using GPU...
        if(self.bUseGpu):

            # Call GPU variant
            self.TrainAutoencoderGPU(raaX, oOptions)

        else:

            # Call CPU variant
            self.TrainAutoencoderCPU(raaX, oOptions)

    def TrainAutoencoderCPU(self, raaX, oOptions):

        # Measure the input vector
        iSamples = raaX.shape[0]
                 
        # For each layer pair...
        for iLayer in range(len(self.oaLayer)):

            # Clone layer weights on device
            raaW = self.oaLayer[iLayer].raaW
            raV  = self.oaLayer[iLayer].raV
            raH  = self.oaLayer[iLayer].raH

            # Measure this layer
            iVs = self.oaLayer[iLayer].iV
            iHs = self.oaLayer[iLayer].iH

            # Create a delta array to retain momentum state
            raaDelta = numpy.zeros((iVs,iHs))
            raDeltaV = numpy.zeros((iVs,1))
            raDeltaH = numpy.zeros((iHs,1))

            # Create a diff array to retain current update
            raaDiff  = numpy.empty((iVs,iHs))
            raDiffV  = numpy.empty((1,iVs))
            raDiffH  = numpy.empty((1,iHs))
            
            # Create an array to retain the layer output for 
            # training the next layer
            raaY = numpy.empty((iSamples, iHs))
            
            # Get short references to layer parameters
            sActivationUp = self.oaLayer[iLayer].sActivationUp
            sActivationDn = self.oaLayer[iLayer].sActivationDn

            junk = None;
            
            # For each training epoch...
            for iEpoch in range(oOptions.iEpochs):

                # Get short references to epoch parameters
                (rRate, rMomentum, rDropV, rDropH, bSample) = oOptions.fTrainingParameters(iLayer,iEpoch)

                # Clear the sample index
                iIndex   = 0
                
                # Clear error accumulators for this layer
                rTotalSe = 0
                rTotalE  = 0

                # While training samples remain...
                while (iIndex<iSamples):

                    # Number of samples to process in this batch
                    iBatch = min(self.iBatchSamples, iSamples-iIndex)

                    # Create working arrays on the device
                    baaH   = numpy.empty((iBatch,iHs))
                    raaH1d = numpy.empty((iBatch,iHs))
                    raaH1s = numpy.empty((iBatch,iHs))
                    raaH3  = numpy.empty((iBatch,iHs))

                    baaV   = numpy.empty((iBatch,iVs))
                    raaV0  = numpy.empty((iBatch,iVs))
                    raaV2  = numpy.empty((iBatch,iVs))

                    # Get a batch of inputs in raaV0
                    # raaX.get_row_slice(iIndex, iIndex+iBatch, target=raaV0)
                    raaV0 = raaX[iIndex:iIndex+iBatch,:]
                    
                    # If we need to drop visible units...
                    if(rDropV>0):
                    
                        # Compute a mask
                        baaV = numpy.random.random(raaV0.shape)<rDropV

                        # Clear dropped states
                        raaV0[baaV] = 0

                    # Advance the markov chain V0->H1
                    raaH1d, raaH1s = self.UpdateStatesCPU(sActivationUp, raaW, raH, raaV0, rDropV, True)

                    # If stochastic sampling is enabled...
                    if (bSample):

                        # Use sampled states
                        raaH1 = raaH1s

                    else:

                        # Use deterministic states
                        raaH1 = raaH1d

                    # If we need to drop hidden units...
                    if(rDropH>0):
                        
                        # Compute a mask
                        baaH = numpy.random.random(raaH1.shape)<rDropH

                        # Clear dropped states
                        raaH1[baaH] = 0

                    # Advance the markov chain H1->V2
                    raaV2, junk  = self.UpdateStatesCPU(sActivationDn, raaW.T, raV, raaH1, rDropH)

                    # If we need to drop visible units...
                    if(rDropV>0):
                        
                        # Clear dropped states
                        raaV2[baaV] = 0

                    # Advance the markov chain V2->H3
                    raaH3, junk  = self.UpdateStatesCPU(sActivationUp, raaW, raH, raaV2, rDropV)

                    # If we need to drop hidden units...
                    if(rDropH>0):
                        
                        # Clear dropped states
                        raaH3[baaH] = 0

                    # Scale factor to average this batch
                    rScale = 1/iBatch
                    
                    # If normalizing the dropout gradient by the number of
                    # weight updates rather the number of batch samples.
                    if (self.bNormalizeDropoutGradient):
                        
                        # If no visible layer dropout...
                        if (not rDropV):
                            
                            # Construct a null dropout matrix
                            baaV[:] = True
                        
                        # If no hidden layer dropout...
                        if (not rDropH):
                            
                            # Construct a null dropout matrix 
                            baaH[:] = True   
                        
                        # Compute normalizer matrix
                        raaN = 1./(double(~baaV).T*(~baaH))

                        # Compute the average difference between positive phase 
                        # up(0,1) and negative phase up(2,3) correlations
                        raaDiff = numpy.multiply( numpy.dot(raaV0.T,raaH1) - numpy.dot(raaV2.T,raaH3) , raaN)

                    else:
                        
                        # Scale all weights uniformly
                        raaDiff = ( numpy.dot(raaV0.T,raaH1) - numpy.dot(raaV2.T,raaH3) )*rScale 

                    # Compute bias gradients
                    raDiffV = numpy.sum(raaV0-raaV2,axis=0)*rScale              
                    raDiffH = numpy.sum(raaH1-raaH3,axis=0)*rScale

                    # Update the weight delta array using the current momentum and
                    # learning rate
                    raaDelta = raaDelta*rMomentum + raaDiff*rRate

                    # Update the weights
                    raaW = raaW + raaDelta
                    
                    # Advance to the next minibatch
                    iIndex = iIndex + iBatch

                # Compute hidden layer
                raaY, junk = self.UpdateStatesCPU(sActivationUp, raaW, raH, raaX)
                
                # Reconstruct visible layer
                raaXr, junk = self.UpdateStatesCPU(sActivationDn, raaW.T, raV, raaY)

                # Compute error metrics
                rTotalSe, rTotalE = self.GetErrorsCPU(raaX, raaXr, sActivationDn)
                    
                # Finish the rmse calculation
                rRmse = math.sqrt(rTotalSe/(raaX.size))
                
                # Finish rmse calculation
                rError = rTotalE/(raaX.size)

                # Report training progress
                oOptions.fEpochReport(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse)
            
            # Current layer outputs are the next layer inputs
            raaX = raaY

            # Save the updated layer parameters
            self.oaLayer[iLayer].raaW = raaW
            self.oaLayer[iLayer].raV  = raV
            self.oaLayer[iLayer].raH  = raH

    def TrainAutoencoderGPU(self, _raaX, oOptions):

        # Copy to device
        raaX = cudamat.CUDAMatrix(_raaX)

        # Count the number of training samples
        iSamples = raaX.shape[0]
                  
        # For each layer pair...
        for iLayer in range(len(self.oaLayer)):

            # Clone layer weights on device
            raaW = cudamat.CUDAMatrix(self.oaLayer[iLayer].raaW)
            raV  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raV))
            raH  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raH))

            # Measure this layer
            iVs = self.oaLayer[iLayer].iV
            iHs = self.oaLayer[iLayer].iH

            # Create a delta array to retain momentum state
            raaDelta = cudamat.zeros((iVs,iHs))
            raDeltaV = cudamat.zeros((iVs,1))
            raDeltaH = cudamat.zeros((iHs,1))

            # Create a diff array to retain current update
            raaDiff  = cudamat.empty((iVs,iHs))
            raDiffV  = cudamat.empty((1,iVs))
            raDiffH  = cudamat.empty((1,iHs))
            
            # Create an array to retain the layer output for 
            # training the next layer
            raaY = cudamat.empty((iSamples, iHs))
            
            # Get short references to layer parameters
            sActivationUp = self.oaLayer[iLayer].sActivationUp
            sActivationDn = self.oaLayer[iLayer].sActivationDn

            junk = None;
            
            # For each training epoch...
            for iEpoch in range(oOptions.iEpochs):

                # Get short references to epoch parameters
                (rRate,rMomentum,rDropV,rDropH,bSample) = oOptions.fTrainingParameters(iLayer,iEpoch)

                # Clear the sample index
                iIndex   = 0
                
                # Clear error accumulators for this layer
                rTotalSe = 0
                rTotalE  = 0

                # While training samples remain...
                while (iIndex<iSamples):

                    # Number of samples to process in this batch
                    iBatch = min(self.iBatchSamples, iSamples-iIndex)

                    # Create working arrays on the device
                    baaH   = cudamat.empty((iBatch,iHs))
                    raaH1d = cudamat.empty((iBatch,iHs))
                    raaH1s = cudamat.empty((iBatch,iHs))
                    raaH3  = cudamat.empty((iBatch,iHs))

                    baaV   = cudamat.empty((iBatch,iVs))
                    raaV0  = cudamat.empty((iBatch,iVs))
                    raaV2  = cudamat.empty((iBatch,iVs))

                    # Get a batch of inputs in raaV0
                    raaX.get_row_slice(iIndex, iIndex+iBatch, target=raaV0)
                    
                    # If we need to drop visible units...
                    if(rDropV>0):
                    
                        # Compute a mask
                        baaV.fill_with_rand()
                        baaV.greater_than(rDropV)
                        raaV0.mult(baaV)

                    # Advance the markov chain V0->H1
                    # raaH1d, raaH1s = self.UpdateStatesGPU(sActivationUp, raaW, raH, raaV0, rDropV, True)
                    self.UpdateStatesGPU(sActivationUp, raaW, raH, raaV0, raaH1d, raaH1s, rDropV, True)

                    # If stochastic sampling is enabled...
                    if (bSample):

                        # Use sampled states
                        raaH1 = raaH1s

                    else:

                        # Use deterministic states
                        raaH1 = raaH1d

                    # If we need to drop hidden units...
                    if(rDropH>0):
                        
                        # Compute a mask
                        baaH.fill_with_rand()
                        baaH.greater_than(rDropH)
                        raaH1.mult(baaH)

                    # Advance the markov chain H1->V2
                    # raaV2, junk  = self.UpdateStatesGPU(sActivationDn, raaW.T, raV, raaH1, rDropH)
                    self.UpdateStatesGPU(sActivationDn, raaW.T, raV, raaH1, raaV2, junk, rDropH)

                    # If we need to drop visible units...
                    if(rDropV>0):
                        
                        # Clear dropped states
                        raaV2.mult(baaV)

                    # Advance the markov chain V2->H3
                    # raaH3, junk  = self.UpdateStatesGPU(sActivationUp, raaW, raH, raaV2, rDropV)
                    self.UpdateStatesGPU(sActivationUp, raaW, raH, raaV2, raaH3, junk, rDropV)

                    # If we need to drop hidden units...
                    if(rDropH>0):
                        
                        # Clear dropped states
                        raaH3.mult(baaH)

                    # Scale factor to average this batch
                    rScale = 1/iBatch
                    
                    # If normalizing the dropout gradient by the number
                    # of weight updates rather the number of batch
                    # samples.
                    if (self.bNormalizeDropoutGradient):
                        
                        # If no visible layer dropout...
                        if (not rDropV):
                            
                            # Construct a null dropout matrix
                            baaV.assign(1)
                        
                        # If no hidden layer dropout...
                        if (not rDropH):
                            
                            # Construct a null dropout matrix 
                            baaH.assign(1)   
                        
                        # Compute normalizer matrix
                        #raaN = 1./(double(~baaV).T*(~baaH))
                        cudamat.dot(baaV.T,baaH,raaN)
                        raaN.reciprocal()

                        # Compute the average difference between positive phase 
                        # up(0,1) and negative phase up(2,3) correlations
                        # raaDiff = numpy.multiply( numpy.dot(raaV0.T,raaH1) - numpy.dot(raaV2.T,raaH3) , raaN)
                        cudamat.dot(raaV0.T,raaH1,raaDiff)
                        raaDiff.subtract_dot(raaV2.T,raaH3)
                        raaDiff.mult(raaN)

                    else:
                        
                        # Scale all weights uniformly
                        #raaDiff = ( numpy.dot(raaV0.T,raaH1) - numpy.dot(raaV2.T,raaH3) )*rScale 
                        cudamat.dot(raaV0.T,raaH1,raaDiff)
                        raaDiff.subtract_dot(raaV2.T,raaH3)
                        raaDiff.mult(rScale)

                    # Compute bias gradients
                    #raDiffV = numpy.sum(raaV0-raaV2,axis=0)*rScale              
                    #raDiffH = numpy.sum(raaH1-raaH3,axis=0)*rScale
                    raaV0.sum(axis=0,mult=rScale).subtract(raaV2.sum(axis=0,mult=rScale),target=raDiffV)
                    raaH1.sum(axis=0,mult=rScale).subtract(raaH3.sum(axis=0,mult=rScale),target=raDiffH)

                    # Update the weight delta array using the current momentum and
                    # learning rate
                    # raaDelta = raaDelta*rMomentum + raaDiff*rRate
                    raaDelta.mult(rMomentum)
                    raaDiff.mult(rRate)
                    raaDelta.add(raaDiff)

                    # Updated the weights
                    #self.oaLayer[iLayer].raaW = self.oaLayer[iLayer].raaW + raaDelta
                    raaW.add(raaDelta)
                    
                    # Advance to the next minibatch
                    iIndex = iIndex + iBatch

                # Create storage for reconstuction
                raaXr = cudamat.empty((iSamples, iVs))

                # Compute hidden layer
                self.UpdateStatesGPU(sActivationUp, raaW, raH, raaX, raaY, junk)

                # Compute visible layer
                self.UpdateStatesGPU(sActivationDn, raaW.T, raV, raaY, raaXr, junk)

                # Compute error metrics
                rTotalSe, rTotalE = self.GetErrorsGPU(raaX, raaXr, sActivationDn)
                    
                # Finish the rmse calculation
                rRmse = math.sqrt(rTotalSe/(raaX.shape[0]*raaX.shape[1]))
                
                # Finish rmse calculation
                rError = rTotalE/(raaX.shape[0]*raaX.shape[1])

                # Report training progress
                oOptions.fEpochReport(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse)
            
            # Current layer outputs are the next layer inputs
            raaX = raaY

            self.oaLayer[iLayer].raaW = raaW.asarray()
            self.oaLayer[iLayer].raV  = raV.asarray()
            self.oaLayer[iLayer].raH  = raH.asarray()

    ## UpdateStates
    # Given an activation type, a weight matrix, and a set of input 
    # layer states, return the logistic output layer states and
    # stochastic states.
    #
    # * sType - specifies the activation def
    # * raaW - specifies the weight matrix
    # * raaX - specifies the input states
    # * raaY - returns the output layer states
    # * baaY - returns the stochastic output states
    
    def UpdateStatesCPU(self, sType, raaW, raB, raaX, rDropout=0, bSample=False):
       
        baaY = [];

        # Compute the scale factor to compensate for dropout so that
        # average activations remain the same
        rScale = 1/(1-rDropout)
        
        # Compute activations
        iRows = raaX.shape[0]

        raaA = numpy.dot(raaX*rScale, raaW)

        for iRow in range(iRows):
            raaA[iRow,:] = raaA[iRow,:] + raB
            
        # Depending on the activation type...
        if (sType=="Logistic"):

            # Compute the logistic def
            raaY = 1./(1+numpy.exp(-raaA))

        elif (sType=="Linear"):

            # Compute output layer states
            raaY = raaA

        elif (sType=="HyperbolicTangent"):

            # Compute output layer states
            raaY = numpy.tanh(raaA)
                                          
        # If stochastic binary states are required...
        if (bSample):

            # Depending on the activation type...
            if (sType=="Logistic"):

                # Sample output layer states
                baaY = raaY > numpy.random.random(raaY.shape)

            elif (sType=="Linear"):

                # Sample output layer states
                baaY = raaY + numpy.random.standard_normal(raaY.shape)

            elif (sType=="HyperbolicTangent"):

                # Sample output layer states
                baaY = raaY > 2*numpy.random.random(raaY.shape)-1

        return(raaY, baaY)

    def UpdateStatesGPU(self, sType, _raaW, _raaB, _raaX, _raaY, _baaY, rDropout=0, bSample=False):

        # Compute the scale factor to compensate for dropout so that
        # average activations remain the same
        rScale = 1/(1-rDropout)
        
        # Compute activations
        cudamat.dot(_raaX, _raaW, target=_raaY)
        _raaY = _raaY.mult(rScale)
        _raaY.add_row_vec(_raaB)
            
        # Depending on the activation type...
        if (sType=="Logistic"):

            # Compute the logistic function
            _raaY.apply_sigmoid(_raaY)

        elif (sType=="Linear"):

            # Compute output layer states
            pass

        elif (sType=="HyperbolicTangent"):

            # Compute output layer states
            _raaY.apply_tanh(_raaY)
                                          
        # If stochastic binary states are required...
        if(bSample):

            # Depending on the activation type...
            if (sType=="Logistic"):

                # Sample output layer states
                _baaY.fill_with_rand()
                _baaY.less_than(_raaY)

            elif (sType=="Linear"):

                # Sample output layer states
                _baaY.fill_with_randn()
                _baaY.add(_raaY)

            elif (sType=="HyperbolicTangent"):

                # Sample output layer states
                _baaY.fill_with_rand()
                _baaY.mult(2)
                _baaY.sub(1)
                _baaY.less_than(_raaY)

    ## Autoencode
    # For each specified sample, autoencode the sample as follows.
    # From the bottom to the top layer, compute the output distribution
    # for the machine given its input. From the top to the bottom
    # layer, Compute the input distribution for the layer given its
    # output. Return the resulting autoencoded sample.
    #
    # Note that the while loop batch processing below is a memory
    # optimization I needed simply because 32 bit matlab ran short of
    # memory when processing MNIST in one chunk. In general these
    # should not be ported for 64 bit use or even for normal 32 bit use
    # since they are ugly and the savings is modest.
    #
    # * raaX - specifies and returns patterns, one pattern per row
    
    def Autoencode(self, raaX):

        # If using GPU...
        if(self.bUseGpu):

            # Call GPU variant
            return(self.AutoencodeGPU(raaX))

        else:

            # Call CPU variant
            return(self.AutoencodeCPU(raaX))

    def AutoencodeCPU(self, raaX):
 
        # Maximum number of patterns to process at once
        iBatch = 1000
        
        # Measure the training data
        iSamples = raaX.shape[0]

        raaXr = numpy.empty((raaX.shape[0],raaX.shape[1]))

        # Clear the index
        iIndex = 0
        
        # While training samples remain...
        while(iIndex<iSamples):
       
            # Get a batch of inputs in raaV0
            raaB = raaX[iIndex:iIndex+iBatch,:]

            junk = [];

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)):

                # Clone layer weights on device
                raaW = self.oaLayer[iLayer].raaW
                raV  = self.oaLayer[iLayer].raV
                raH  = self.oaLayer[iLayer].raH

                # Measure this layer
                iVs = self.oaLayer[iLayer].raaW.shape[0]
                iHs = self.oaLayer[iLayer].raaW.shape[1]

                # Create an array to retain the layer output for 
                # training the next layer
                raaY = numpy.empty((iBatch, iHs))                

                # Get short references to layer parameters
                sActivationUp = self.oaLayer[iLayer].sActivationUp
                sActivationDn = self.oaLayer[iLayer].sActivationDn

                raaY, junk = self.UpdateStatesCPU(sActivationUp, raaW, raH, raaB)

                raaB = raaY

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)-1, -1, -1):

                # Clone layer weights on device
                raaW = self.oaLayer[iLayer].raaW
                raV  = self.oaLayer[iLayer].raV
                raH  = self.oaLayer[iLayer].raH

                # Measure this layer
                iVs = self.oaLayer[iLayer].raaW.shape[0]
                iHs = self.oaLayer[iLayer].raaW.shape[1]

                # Create an array to retain the layer output for 
                # training the next layer
                raaY = numpy.empty((iBatch, iVs))

                # Get short references to layer parameters
                sActivationUp = self.oaLayer[iLayer].sActivationUp
                sActivationDn = self.oaLayer[iLayer].sActivationDn

                raaY, junk = self.UpdateStatesCPU(sActivationDn, raaW.T, raV, raaB)

                raaB = raaY
            
            # Save reconstruction states
            raaXr[iIndex:iIndex+iBatch,:] = raaB
            
            # Advance to the next batch
            iIndex = iIndex + iBatch

        # Return reconstruction
        return(raaXr)

    def AutoencodeGPU(self, _raaX):
        
        # Maximum number of patterns to process at once
        iBatch = 1000
        
        # Measure the training data
        iSamples = _raaX.shape[0]
        
        raaX = cudamat.CUDAMatrix(_raaX)

        # Clear the index
        iIndex = 0
        
        # While training samples remain...
        while(iIndex<iSamples):

            # Compute largest allowable batch size
            iBatch = min(iBatch, iSamples-iIndex)

            # Compute an indexer
            raaB = cudamat.empty((iBatch,raaX.shape[1]))
            
            # Get a batch of inputs in raaV0
            raaX.get_row_slice(iIndex, iIndex+iBatch, target=raaB)

            junk = []

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)):

                # Clone layer weights on device
                raaW = cudamat.CUDAMatrix(self.oaLayer[iLayer].raaW)
                raV  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raV))
                raH  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raH))

                # Measure this layer
                iVs = self.oaLayer[iLayer].raaW.shape[0]
                iHs = self.oaLayer[iLayer].raaW.shape[1]

                # Create an array to retain the layer output for 
                # training the next layer
                raaY = cudamat.empty((iBatch, iHs))                

                # Get short references to layer parameters
                sActivationUp = self.oaLayer[iLayer].sActivationUp
                sActivationDn = self.oaLayer[iLayer].sActivationDn

                self.UpdateStatesGPU(sActivationUp, raaW, raH, raaB, raaY, junk)

                raaB = raaY

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)-1, -1, -1):

                # Clone layer weights on device
                raaW = cudamat.CUDAMatrix(self.oaLayer[iLayer].raaW)
                raV  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raV))
                raH  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raH))

                # Measure this layer
                iVs = self.oaLayer[iLayer].raaW.shape[0]
                iHs = self.oaLayer[iLayer].raaW.shape[1]

                # Create an array to retain the layer output for 
                # training the next layer
                raaY = cudamat.empty((iBatch, iVs))

                # Get short references to layer parameters
                sActivationUp = self.oaLayer[iLayer].sActivationUp
                sActivationDn = self.oaLayer[iLayer].sActivationDn

                self.UpdateStatesGPU(sActivationDn, raaW.T, raV, raaB, raaY, junk)

                raaB = raaY
            
            # Save reconstruction states
            raaX.set_row_slice(iIndex, iIndex+iBatch, raaB)
            
            # Advance to the next batch
            iIndex = iIndex + iBatch

        _raaX = raaX.asarray()

        return(_raaX)

    ## GetErrors
    # Compute the total squared error and the total activation 
    # specific error measure for all specified samples.
    #
    # * raaX - specifies the autoencoder input
    # * raaY - specifies the autoencoder output
    # * rSe - returns the sum of squared error for all input elements
    # * rE - returns the sum of activation specific errors for
    #   all elements

    def GetErrorsCPU(self, raaX, raaY, sActivation):
        
        # Small value to avoid log underflows
        rEps = 1e-20         
        
        # Compute error
        raaError = raaX-raaY

        # Sum all squared errors
        rSe = numpy.sum(numpy.square(raaError))

        # Depending on the activation def type
        if(sActivation=="Logistic"):

            # Compute the cross entropy error
            rE = -numpy.sum(numpy.multiply(raaX,numpy.log(raaY+rEps)) + numpy.multiply(1-raaX,numpy.log(1-raaY+rEps)))

        elif(sActivation=="Linear"):

            # Compute the squared error
            rE = rSe

        elif(sActivation=="Softmax"):

            # Compute the cross entropy error
            rE = -numpy.sum(numpy.multiply(raaX,numpy.log(raaY+rEps)))
          
        return(rSe, rE)

    def GetErrorsGPU(self, raaX, raaY, sActivation):
        
        # Small value to avoid log underflows
        rEps = 1e-20   

        # Create error matrix
        raaError = cudamat.empty(raaX.shape)

        # Compute error
        raaX.subtract(raaY, raaError)

        # Compute sum of squares
        rSe = raaError.euclid_norm()**2    

        # Depending on the activation def type
        if(sActivation=="Logistic"):

            _raaX = raaX.asarray()
            _raaY = raaY.asarray()

            # Compute the cross entropy error
            rE = -numpy.sum(numpy.multiply(_raaX,numpy.log(_raaY+rEps)) + numpy.multiply(1-_raaX,numpy.log(1-_raaY+rEps)))

        elif(sActivation=="Linear"):

            # Compute the squared error
            rE = rSe

        elif(sActivation=="Softmax"):

            _raaX = raaX.asarray()
            _raaY = raaY.asarray()

            # Compute the cross entropy error
            rE = -numpy.sum(numpy.multiply(_raaX,numpy.log(_raaY+rEps)))
          
        return(rSe, rE)

    ## ComputeReconstructionError
    # Autoencode the specified samples and compute error metrics 
    # for the reconstructions.
    # 
    # * raaX - specifies the samples
    # * rRmse - returns the root mean squared error
    # * rError - returns the error measure for the output layer
    #   activation type
    
    def ComputeReconstructionError(self, raaX):
        
        # Autoencode the specified samples to form reconstructions
        raaY = self.Autoencode(raaX)
        
        # Compute total error metrics
        (rTotalSe, rTotalE) = self.GetErrorsCPU(raaX, raaY, self.oaLayer[0].sActivationDn)
            
        # Compute average RMSE
        rRmse = math.sqrt(rTotalSe/raaX.size)

        # Compute average E
        rE = math.sqrt(rTotalE/raaX.size)
              
        return(rRmse, rE)
    
# Define a function to exercise the class (crudely!)
def Test(sSrc):

    # Import supporting libraries
    import pandas

    # Specify epochs
    iEpochs = 10

    # Read the MNIST dataset as a pandas.DataFrame
    df = pandas.read_pickle(sSrc)

    # Retrieve the pixel columns and scale them from zero to one
    raaX = numpy.array(df.ix[:,0:783])/256.0

    # Create 784 x 1000 x 30 rbm layers
    oaLayers = [Layer(raaX.shape[1],1000),Layer(1000,30)]

    # Create training options
    oOptions = Options(iEpochs)

    # Create RbmStack
    oRbmStack = RbmStack(oaLayers, True)

    # Train using the specified options
    oRbmStack.TrainAutoencoder(raaX, oOptions)

    (rRmse, rE) = oRbmStack.ComputeReconstructionError(raaX)

    print("rRmse={:.6f}, rE={:.6f}".format(rRmse,rE))

Test("MNIST.pkl")