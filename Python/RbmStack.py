import numpy

#RbmStack
# This class implements a stack of restricted Boltzmann machines. It 
# provides methods to train the stack using a greedy layerwise
# approach, to perform inference using the stack as an autoencoder,
# and to unroll the stack to initialize a deep feedforward network
# implementing either an autoencoder or a classifier.
#
# The RbmStack provides the following features:
#
# * Allows specification of architectures having varying numbers of
#   layers with specified size and activation defs.
# * Optionally implements dropout of visible and/or hidden neurons
#   during training.
# * Means to specify the learning rate, momentum and dropout 
#   probabilities for each layer during each training epoch.
# * Event callback mechanism for reporting training progress
#   and performance metrics.

class RbmStack:
    
    ## properties
        
    # Number of samples per training batch
    iBatchSamples = 100
    
    # Variance for random weight initialization
    rInitialWeightVariance = 0.100   
    
    # Normalize the dropout gradient by the number of weight updates
    # rather than the number of training samples.
    bNormalizeDropoutGradient = False

    # Layer properties
    # * oaLayer(k).iSize
    # * oaLayer(k).sActivationUp - upward activation   ("Logistic", "Linear", "HyperbolicTangent")
    # * oaLayer(k).sActivationDn - downward activation ("Logistic", "Linear", "HyperbolicTangent")
    # * oaLayer(k).raaW - weights
    oaLayer = []
          
    ## methods
        
    ## RbmStack
    # Construct a new object with the specified configuration. The 
    # configuration object must be a structure array and its length
    # must be one greater than the number of machines in the stack. 
    # Only the iSize parameter is used for the final layer oaLayer(end).
    #
    # * oaLayer - specifies the network configuration.
    
    def __init__(self, oaLayer):

        # Copy the layer properties
        self.oaLayer = oaLayer
          
        # For each weight layer...
        for iLayer in range(len(self.oaLayer)-1):
        
            # Initialize random weights while adding extra biases
            self.oaLayer[iLayer].raaW = numpy.random.randn(self.oaLayer[iLayer+0].iSize, self.oaLayer[iLayer+1].iSize) * self.rInitialWeightVariance

            # Clear the biases
            self.oaLayer[iLayer].raV = numpy.zeros(self.oaLayer[iLayer+0].iSize)
            self.oaLayer[iLayer].raH = numpy.zeros(self.oaLayer[iLayer+1].iSize)
                  
    ## TrainAutoencoder
    # Perform greedy pre-training of the network using the specified
    # training data and options.
    #
    # *raaX - specifies the network training samples
    # *oOptions - specifies training options
    # *oOptions.oaLayer[iLayer].raDropV[iEpoch]
    # *oOptions.oaLayer[iLayer].raDropH[iEpoch]
    # *oOptions.oaLayer[iLayer].raMomentum[iEpoch]
    # *oOptions.oaLayer[iLayer].raRate[iEpoch]
    # *oOptions.oaLayer[iLayer].baSample[iEpoch]

    def TrainAutoencoder(self, raaX, oOptions):
     
        # Count the number of training samples
        iSamples = raaX.shape[0]
                  
        # For each layer pair...
        for iLayer in range(len(self.oaLayer)-1):

            # Create a delta array to retain momentum state
            raaDelta = numpy.zeros(self.oaLayer[iLayer].raaW.shape)
            raDeltaV = numpy.zeros(self.oaLayer[iLayer].raV.shape)
            raDeltaH = numpy.zeros(self.oaLayer[iLayer].raH.shape)

            # Create a diff array to retain current update
            raaDiff = numpy.zeros(self.oaLayer[iLayer].raaW.shape)
            raaDiffV = numpy.zeros(self.oaLayer[iLayer].raV.shape)
            raaDiffH = numpy.zeros(self.oaLayer[iLayer].raH.shape)
            
            # Create an array to retain the layer output for 
            # training the next layer
            raaY = numpy.zeros((iSamples, self.oaLayer[iLayer].raaW.shape[1]))
            
            # Get short references to layer parameters
            sActivationUp = self.oaLayer[iLayer].sActivationUp
            sActivationDn = self.oaLayer[iLayer].sActivationDn
            
            # For each training epoch...
            for iEpoch in range(oOptions.iEpochs):

                # Get short references to epoch parameters
                rDropV    = oOptions.oaLayer[iLayer].raDropV[iEpoch]
                rDropH    = oOptions.oaLayer[iLayer].raDropH[iEpoch]
                rMomentum = oOptions.oaLayer[iLayer].raMomentum[iEpoch]
                rRate     = oOptions.oaLayer[iLayer].raRate[iEpoch]
                bSample   = oOptions.oaLayer[iLayer].baSample[iEpoch]

                # Clear the sample index
                iIndex = 0
                
                # Clear error accumulators for this layer
                rTotalSe = 0
                rTotalE  = 0

                # While training samples remain...
                while (iIndex<iSamples):

                    # Compute an indexer
                    ia = range(iIndex,min(iIndex+self.iBatchSamples,iSamples))

                    # Get a batch of inputs
                    raaV0 = numpy.copy(raaX[ia,:])
                    
                    # If we need to drop visible units...
                    if (rDropV>0):
                    
                        # Compute a mask
                        baaV = numpy.random.random(raaV0.shape)<rDropV

                        # Clear dropped states
                        raaV0[baaV] = 0
                        
                    # Advance the markov chain V0->H1
                    raaH1d, raaH1s = self.UpdateStates(sActivationUp, self.oaLayer[iLayer].raaW, self.oaLayer[iLayer].raH, raaV0, rDropV, True)

                    # If stochastic sampling is enabled...
                    if (bSample):

                        # Use sampled states
                        raaH1 = raaH1s

                    else:

                        # Use deterministic states
                        raaH1 = raaH1d

                    # If we need to drop hidden units...
                    if (rDropH>0):
                        
                        # Compute a mask
                        baaH = numpy.random.random(raaH1.shape) < rDropH

                        # Clear dropped states
                        raaH1[baaH] = 0

                    # Advance the markov chain H1->V2
                    raaV2, junk  = self.UpdateStates(sActivationDn, self.oaLayer[iLayer].raaW.T, self.oaLayer[iLayer].raV, raaH1, rDropH)

                    # If we need to drop visible units...
                    if (rDropV>0):
                        
                        # Clear dropped states
                        raaV2[baaV] = 0

                    # Advance the markov chain V2->H3
                    raaH3, junk  = self.UpdateStates(sActivationUp, self.oaLayer[iLayer].raaW, self.oaLayer[iLayer].raH, raaV2, rDropV)


                    # If we need to drop hidden units...
                    if (rDropH>0):
                        
                        # Clear dropped states
                        raaH3[baaH] = 0

                    # Scale factor to average this batch
                    rScale = 1/len(ia)
                    
                    # If normalizing the dropout gradient by the number
                    # of weight updates rather the number of batch
                    # samples.
                    if (self.bNormalizeDropoutGradient):
                        
                        # If no visible layer dropout...
                        if (not rDropV):
                            
                            # Construct a null dropout matrix
                            baaV = numpy.zeros(raaV0.shape)
                        
                        # If no hidden layer dropout...
                        if (not rDropH):
                            
                            # Construct a null dropout matrix 
                            baaH = numpy.zeros(raaH1.shape)   
                        
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

                    # Updated the weights
                    self.oaLayer[iLayer].raaW = self.oaLayer[iLayer].raaW + raaDelta
                    
                    # If the visible layer used dropout...
                    if (rDropV):
                        
                        # Recompute H1 with no dropout
                        raaY[ia,:], junk = self.UpdateStates(sActivationUp, self.oaLayer[iLayer].raaW, self.oaLayer[iLayer].raH, raaX[ia,:], 0)
                        
                        # Recompute V2 based on the new H1
                        raaV2, junk = self.UpdateStates(sActivationDn, self.oaLayer[iLayer].raaW.T, self.oaLayer[iLayer].raV, raaY[ia,:], 0) 
                        
                    else:
                        
                        # Use the prior computation
                        raaY[ia,:] = raaH1d
                        
                        # If the hidden layer used dropout or sampling...
                        if (rDropH or bSample):
                            
                            # Recompute V2
                            raaV2, junk = self.UpdateStates(sActivationDn, self.oaLayer[iLayer].raaW.T, self.oaLayer[iLayer].raV, raaY[ia,:], 0) 
                    
                    # Gather error statistics for this minibatch
                    rSe, rE = self.GetErrors(raaX[ia,:], raaV2, sActivationDn)
                    
                    # Accumulate total errors
                    rTotalSe = rTotalSe+rSe
                    rTotalE  = rTotalE + rE
                    
                    # Advance to the next minibatch
                    iIndex = iIndex + len(ia)
                
                # Finish the rmse calculation
                rRmse = numpy.sqrt(rTotalSe/raaX.size)
                
                # Record the error for this epoch
                self.oaLayer[iLayer].raRmse[iEpoch] = rRmse 
                
                # Finish rmse calculation
                rError = rTotalE/raaX.size
                
                # Record the error for this epoch
                self.oaLayer[iLayer].raError[iEpoch] = rError

                # Report training progress
                # oOptions.fEvent(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError)
                print(rRmse)
            
            # Current layer outputs are the next layer inputs
            raaX = raaY
                     
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
    
    def UpdateStates(self, sType, raaW, raB, raaX, rDropout=0, bSample=False):
       
        baaY = [];

        # Compute the scale factor to compensate for dropout so that
        # average activations remain the same
        rScale = 1/(1-rDropout)
        
        # Compute activations
        iRows = raaX.shape[0]
        raaA = numpy.dot(raaX*rScale, raaW)

        for iRow in range(iRows):
            raaA[iRow,:] += raB
            
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

    def _TrainAutoencoder(self, raaX, oOptions):
     
        # initialize cudamat
        cudamat.cublas_init()
        cudamat.CUDAMatrix.init_random(seed = 42)

        _raaX = cudamat.CUDAMatrix(raaX)
        #print(_raaX.asarray())

        # Count the number of training samples
        iSamples = raaX.shape[0]
                  
        # For each layer pair...
        for iLayer in range(len(self.oaLayer)-1):

            # Create a delta array to retain momementum state
            raaDelta = numpy.zeros(self.oaLayer[iLayer].raaW.shape)
            _raaDelta = cudamat.CUDAMatrix(raaDelta)

            # Create a diff array to retain current update
            raaDiff = numpy.zeros(self.oaLayer[iLayer].raaW.shape)
            _raaDiff = cudamat.CUDAMatrix(raaDiff)
            
            # Create an array to retain the layer output for 
            # training the next layer
            raaY = numpy.zeros((iSamples, self.oaLayer[iLayer].raaW.shape[1]-1))
            _raaY = cudamat.CUDAMatrix(raaY)
            
            # Get short references to layer parameters
            sActivationUp = self.oaLayer[iLayer].sActivationUp
            sActivationDn = self.oaLayer[iLayer].sActivationDn
            
            # For each training epoch...
            for iEpoch in range(oOptions.iEpochs):

                # Get short references to epoch parameters
                rDropV    = oOptions.oaLayer[iLayer].raDropV[iEpoch]
                rDropH    = oOptions.oaLayer[iLayer].raDropH[iEpoch]
                rMomentum = oOptions.oaLayer[iLayer].raMomentum[iEpoch]
                rRate     = oOptions.oaLayer[iLayer].raRate[iEpoch]
                bSample   = oOptions.oaLayer[iLayer].baSample[iEpoch]

                # Clear the sample index
                iIndex = 0
                
                # Clear error accumulators for this layer
                rTotalSe = 0
                rTotalE  = 0

                _raaW = cudamat.CUDAMatrix(self.oaLayer[iLayer].raaW)

                # While training samples remain...
                while (iIndex<iSamples):

                    # Compute an indexer
                    ia = range(iIndex,min(iIndex+self.iBatchSamples,iSamples))

                    # Get a batch of inputs
                    #raaV0 = numpy.copy(raaX[ia,:])
                    _raaV0 = _raaX.get_row_slice(iIndex,min(iIndex+self.iBatchSamples,iSamples))
                    
                    # If we need to drop visible units...
                    if (rDropV>0):
                    
                        # Compute a mask
                        #baaV = numpy.random.random(raaV0.shape)<rDropV
                        _baaV = cudamat.empty(_raaV0.shape)
                        _baaV.fill_with_rand()
                        _baaV.greater_than(rDropV)
                        
                        # Clear dropped states
                        #raaV0[baaV] = 0
                        _raaV0.mult(_baaV)
                        
                    # Advance the markov chain V0->H1
                    _raaH1d, _raaH1s = self._UpdateStates(sActivationUp, _raaW, _raaV0, rDropV, True)
                    #raaH1d, raaH1s = self.UpdateStates(sActivationUp, self.oaLayer[iLayer].raaW, raaV0, rDropV, True)

                    #print(raaH1d-_raaH1d)

                    # If stochastic sampling is enabled...
                    if (bSample):

                        # Use sampled states
                        #raaH1 = raaH1s
                        _raaH1 = _raaH1s

                    else:

                        # Use deterministic states
                        #raaH1 = raaH1d
                        _raaH1 = _raaH1d

                    # If we need to drop hidden units...
                    if (rDropH>0):
                        
                        # Compute a mask
                        #baaH = numpy.random.random(raaH1.shape) < rDropH
                        _baaH = cudamat.empty(_raaH0.shape)
                        _baaH.fill_with_rand()
                        _baaH.greater_than(rDropH)

                        # Clear dropped states
                        #raaH1[baaH] = 0
                        _raaH1.mult(_baaH)

                    # Advance the markov chain H1->V2
                    #raaV2, junk  = self._UpdateStates(sActivationDn, self.oaLayer[iLayer].raaW.T, raaH1, rDropH)
                    _raaV2, junk  = self._UpdateStates(sActivationDn, _raaW.T, _raaH1, rDropH)

                    # If we need to drop visible units...
                    if (rDropV>0):
                        
                        # Clear dropped states
                        raaV2[baaV] = 0

                    # Advance the markov chain V2->H3
                    #raaH3, junk  = self._UpdateStates(sActivationUp, self.oaLayer[iLayer].raaW,  raaV2, rDropV)
                    _raaH3, junk  = self._UpdateStates(sActivationUp, _raaW,  _raaV2, rDropV)

                    # If we need to drop hidden units...
                    if (rDropH>0):
                        
                        # Clear dropped states
                        raaH3[baaH] = 0

                    # Scale factor to average this batch
                    rScale = 1/len(ia)
                    
                    # If normalizing the dropout gradient by the number
                    # of weight updates rather the number of batch
                    # samples.
                    if (self.bNormalizeDropoutGradient):
                        
                        # If no visible layer dropout...
                        if (not rDropV):
                            
                            # Construct a null dropout matrix
                            baaV = numpy.zeros(raaV0.shape)
                        
                        # If no hidden layer dropout...
                        if (not rDropH):
                            
                            # Construct a null dropout matrix 
                            baaH = numpy.zeros(raaH1.shape)   
                        
                        # Compute normalizer matrix
                        raaN = 1./(double(~baaV).T*(~baaH))

                        # Compute the average difference between positive phase 
                        # up(0,1) and negative phase up(2,3) correlations
                        raaDiff[:-1,:-1] = numpy.multiply( numpy.dot(raaV0.T,raaH1) - numpy.dot(raaV2.T,raaH3) , raaN)
                        
                    else:
                        
                        # Scale all weights uniformly
                        raaDiff[:-1,:-1] = ( numpy.dot(raaV0.T,raaH1) - numpy.dot(raaV2.T,raaH3) )*rScale 
                      
                    # Compute bias gradients
                    raDiffV = numpy.sum(raaV0-raaV2,axis=0)*rScale              
                    raDiffH = numpy.sum(raaH1-raaH3,axis=0)*rScale

                    # Augment weight differences with biases
                    raaDiff[-1,:-1] = raDiffH
                    raaDiff[:-1,-1] = raDiffV.T

                    # Update the weight delta array using the current momentum and
                    # learning rate
                    raaDelta = raaDelta*rMomentum + raaDiff*rRate

                    # Updated the weights
                    self.oaLayer[iLayer].raaW = self.oaLayer[iLayer].raaW + raaDelta
                    
                    # If the visible layer used dropout...
                    if (rDropV):
                        
                        # Recompute H1 with no dropout
                        raaY[ia,:], junk = self._UpdateStates(sActivationUp, self.oaLayer[iLayer].raaW, raaX[ia,:], 0)
                        
                        # Recompute V2 based on the new H1
                        raaV2, junk = self._UpdateStates(sActivationDn, self.oaLayer[iLayer].raaW.T, raaY[ia,:], 0) 
                        
                    else:
                        
                        # Use the prior computation
                        raaY[ia,:] = raaH1d
                        
                        # If the hidden layer used dropout or sampling...
                        if (rDropH or bSample):
                            
                            # Recompute V2
                            raaV2, junk = self._UpdateStates(sActivationDn, self.oaLayer[iLayer].raaW.T, raaY[ia,:], 0) 
                    
                    # Gather error statistics for this minibatch
                    rSe, rE = self.GetErrors(raaX[ia,:], raaV2, sActivationDn)
                    
                    # Accumulate total errors
                    rTotalSe = rTotalSe+rSe
                    rTotalE  = rTotalE + rE
                    
                    # Advance to the next minibatch
                    iIndex = iIndex + len(ia)
                
                # Finish the rmse calculation
                rRmse = numpy.sqrt(rTotalSe/raaX.size)
                
                # Record the error for this epoch
                self.oaLayer[iLayer].raRmse[iEpoch] = rRmse 
                
                # Finish rmse calculation
                rError = rTotalE/raaX.size
                
                # Record the error for this epoch
                self.oaLayer[iLayer].raError[iEpoch] = rError

                # Report training progress
                # oOptions.fEvent(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError)
                print(rRmse)
            
            # Current layer outputs are the next layer inputs
            raaX = raaY

    def _UpdateStates(self, sType, _raaW, _raaX, rDropout=0, bSample=False):
       
        _baaY = cudamat.CUDAMatrix(numpy.atleast_2d(0))

        # Compute the scale factor to compensate for dropout so that
        # average activations remain the same
        rScale = 1/(1-rDropout)
        
        # Compute activations
        #iRows = raaX.shape[0]
       # raaA = numpy.dot( numpy.concatenate( (raaX*rScale, numpy.ones(iRows).reshape(iRows,1) ), axis=1), raaW[:,:-1] )

        #_raaA = cudamat.CUDAMatrix(raaA)

        #_raaX = cudamat.empty((raaX.shape[0],raaX.shape[1]+1);
        #_raaX = cudamat.CUDAMatrix(raaX)
        _raaX = _raaX.mult(rScale)
        _raaW = cudamat.CUDAMatrix(raaW[:-1,:-1])

        _raaB = cudamat.CUDAMatrix(numpy.atleast_2d(raaW[-1,:-1]))
        _raaA = _raaX.dot(_raaW)
        _raaA.add_row_vec(_raaB)

        # allocate outputs
        _raaY = cudamat.empty(_raaA.shape)
            
        # Depending on the activation type...
        if (sType=="Logistic"):

            # Compute the logistic def
       #     raaY = 1./(1+numpy.exp(-raaA))
            _raaA.apply_sigmoid(_raaY)

        elif (sType=="Linear"):

            # Compute output layer states
       #     raaY = raaA
            _raaY.assign(_raaY)

        elif (sType=="HyperbolicTangent"):

            # Compute output layer states
        #    raaY = numpy.tanh(raaA)
            _raaY.apply_tanh(_raaY)
                                          
        # If stochastic binary states are required...
        if (bSample):

            _baaY = cudamat.empty(_raaY.shape)

            # Depending on the activation type...
            if (sType=="Logistic"):

                # Sample output layer states
        #        baaY = raaY > numpy.random.random(raaY.shape)
                _baaY.fill_with_rand()
                _baaY.less_than(_raaY)

            elif (sType=="Linear"):

                # Sample output layer states
        #        baaY = raaY + numpy.random.standard_normal(raaY.shape)
                _baaY.fill_with_rand()
                _baaY.add(_raaY)

            elif (sType=="HyperbolicTangent"):

                # Sample output layer states
        #        baaY = raaY > 2*numpy.random.random(raaY.shape)-1
                _baaY.fill_with_rand()
                _baaY.mult(2)
                _baaY.sub(1)
                _baaY.less_than(_raaY)

        return(_raaY.asarray(), _baaY.asarray())
    
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
        
        # Maximum number of patterns to process at once
        iBatch = 1000
        
        # Measure the training data
        iSamples = raaX.shape[0]
        
        # Clear the index
        iIndex = 0
        
        # While training samples remain...
        while(iIndex<iSamples):

            # Compute an indexer
            ia = iIndex+range(numpy.min(iIndex+iBatch,iSamples))
            
            # Extract pattern batch
            raaB = numpy.copy(raaX[ia,:])

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)-1):

                # Propagate states upward
                raaB = UpdateStates(self, self.oaLayer[iLayer].sActivationUp, self.oaLayer[iLayer].raaW, raaB)

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)-2, 0, -1):

                # Propagate states downward:
                raaB = UpdateStates(self, self.oaLayer[iLayer].sActivationDn, self.oaLayer[iLayer].raaW.T, raaB)
            
            # Save reconstruction states
            raaX[ia,:] = raaB
            
            # Advance to the next batch
            iIndex = iIndex + len(ia)

        return(raaX)

    ## ComputeReconstructionError
    # Autoencode the specified samples and compute error metrics 
    # for the reconstructions.
    # 
    # * raaX - specifies the samples
    # * rRmse - returns the root mean squared error
    # * rError - returns the error measure for the output layer
    #   activation type
    
    def ComputeReconstructionError(self, raaX):
        
        # Process small batches to conserve memory
        iBatch = 1000
        
        # Autoencode the specified samples to form reconstructions
        raaY = Autoencode(self, raaX)
        
        # Measure the data
        iSamples = raaX.shape[0]
        
        # Clear the index
        iIndex = 0
        
        # Clear the error accumulators
        rTotalE  = 0
        rTotalSe = 0
        
        # While training samples remain...
        while(iIndex<iSamples):
            
            # Compute an indexer
            ia = iIndex + range(numpy.min(iIndex+iBatch,iSamples))
            
            # Get errors for this batch
            rSe, rE = GetErrors(self, raaX[ia,:], raaY[ia,:], self.oaLayer[0].sActivationDn)
            
            # Accumulate the error totals
            rTotalSe = rTotalSe+rSe
            rTotalE = rTotalE + rE

            # Increment the index
            iIndex = iIndex + leng(ia)
        
        # Average error over all samples
        rError = rTotalE/raaX.size
        
        # Root mean square error over all samples
        rRmse  = numpy.sqrt(rTotalSe/raaX.size)
        
        return(rRmse, rError)
    
    ## GetErrors
    # Compute the total squared error and the total activation def 
    # specific error measure for all specified samples.
    #
    # * raaX - specifies the autoencoder input
    # * raaY - specifies the autoencoder output
    # * rSe - returns the sum of squared error for all input elements
    # * rE - returns the sum of activation def specific errors for
    #   all elements

    def GetErrors(self, raaX, raaY, sActivation):
        
        # Small value to avoid log underflows
        rEps = 1e-80         
        
        # Sum all squared errors
        raaError = raaX-raaY

        rSe = numpy.sum(numpy.multiply(raaError,raaError))
        
        # Depending on the activation def type
        if(sActivation=="Logistic"):

            # Compute the average cross entropy error
            rE = -numpy.sum(numpy.multiply(raaX,numpy.log(raaY+rEps)) + numpy.multiply(1-raaX,numpy.log(1-raaY+rEps)))

        elif(sActivation=="Linear"):

            # Compute the squared error
            rE = rSe

        elif(sActivation=="Softmax"):

            # Compute the average cross entropy error
            rE = -numpy.sum(numpy.multiply(raaX,numpy.log(raaY+rEps)))
            
        return(rSe, rE)
    
    ## Unroll
    # Extract initialization parameters for the specified deep neural 
    # network type.
    #
    # * bAutoencoder - specifies whether to extract an autoencoder or
    #   a classifier (true-autoencoder, false-classifier)
    # * oaLayer - returns the network initialization parameters
    # * oaLayer(k).sActivation - activation type for layer k
    # * oaLayer(k).raaW - weights for layer k
    
    def Unroll(self, bAutoencoder):
        
        # Initialize the output layer index
        k = 0
        
        # For each weight layer... 
        for iLayer in range(len(self.oaLayer)-1):
            
            # Increment the output layer index
            k = k + 1
            
            # Get the weights while ignoring reverse biases
            oaLayer[k].raaW = self.oaLayer[iLayer].raaW[:,:-1]
            
            # Get the type
            oaLayer[k].sActivation = self.oaLayer[iLayer].sActivationUp
        
        # If extracting an autoencoder...
        if (bAutoencoder):
 
            # For each weight layer...
            for iLayer in range(len(self.oaLayer)-1,-1,-1):
                
                # Increment the output layer index
                k = k + 1

                # Get the transposed weights while ignoring forward biases
                oaLayer[k].raaW  = self.oaLayer[iLayer].raaW[:-1,:].T
                
                # Get the type
                oaLayer[k].sActivation = self.oaLayer[iLayer].sActivationDn
            
        return(oaLayer)

    @staticmethod
    def TestX():
        print('Hello')

# Define a function to exercise the class (crudely!)
def Test():

    # Import supporting libraries
    import pandas
    import pprint

    # Define classes used for RbmStack interfacing
    class Layer:

        def __init__(self, iSize, iEpochs):
            self.iSize = iSize
            self.raaW = object
            self.sActivationUp = "Logistic"
            self.sActivationDn = "Logistic"
            self.raRmse     = numpy.zeros(iEpochs)
            self.raError    = numpy.zeros(iEpochs)

    class LayerOptions:

        def __init__(self, iEpochs):

            self.raDropV    = 0.1*numpy.ones(iEpochs)
            self.raDropH    = 0.1*numpy.ones(iEpochs)
            self.raMomentum = 0.9*numpy.zeros(iEpochs)
            self.raMomentum[:5]=0.5;
            self.raRate     = 0.1*numpy.ones(iEpochs)
            self.baSample   = numpy.zeros(iEpochs)
            self.raRmse     = numpy.zeros(iEpochs)


    class Options:

        def __init__(self, iEpochs):

            self.iEpochs = iEpochs
            self.oaLayer = [LayerOptions(iEpochs), LayerOptions(iEpochs)]  

    # Specify epochs
    iEpochs = 10

    # Read the MNIST dataset as a pandas.DataFrame
    df = pandas.read_pickle("../Datasets/MNIST/MNIST.pkl")

    # Retrieve the pixel columns and scale them from zero to one
    raaX = numpy.array(df.ix[:100,0:783])/256.0

    # Create 784 x 1000 x 30 rbm layers
    oaLayers = [Layer(raaX.shape[1],iEpochs),Layer(1000,iEpochs),Layer(30,iEpochs)]

    # Create training options
    oOptions = Options(iEpochs)

    # Create RbmStack
    oRbmStack = RbmStack(oaLayers)

    # Train using the specified options
    oRbmStack.TrainAutoencoder(raaX, oOptions)

    #print(o.oaLayer[1].raaW)

Test()

