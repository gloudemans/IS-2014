# To Do:
# Determine why code still causes a very large number of CUDA matrix initializations
# Update code so that data rows don't need to be a multiple of batch size.
# Determine why autoencoder RMSE performance is lower than matlab code.

import math
import numpy
import cudamat

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

    def TrainAutoencoder(self, _raaX, oOptions):

        # initialize cudamat
        cudamat.init()
        cudamat.CUDAMatrix.init_random(seed = 42)

        # Count the number of training samples
        raaX = cudamat.CUDAMatrix(_raaX)
        iSamples = raaX.shape[0]
                  
        # For each layer pair...
        for iLayer in range(len(self.oaLayer)-1):

            # Clone layer weights on device
            raaW = cudamat.CUDAMatrix(self.oaLayer[iLayer].raaW)
            raV  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raV))
            raH  = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayer[iLayer].raH))

            # Measure this layer
            iVs = self.oaLayer[iLayer].raaW.shape[0]
            iHs = self.oaLayer[iLayer].raaW.shape[1]

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
                rDropV    = oOptions.oaLayer[iLayer].raDropV[iEpoch]
                rDropH    = oOptions.oaLayer[iLayer].raDropH[iEpoch]
                rMomentum = oOptions.oaLayer[iLayer].raMomentum[iEpoch]
                rRate     = oOptions.oaLayer[iLayer].raRate[iEpoch]
                bSample   = oOptions.oaLayer[iLayer].baSample[iEpoch]

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
                    # raaH1d, raaH1s = self._UpdateStates(sActivationUp, raaW, raH, raaV0, rDropV, True)
                    self._UpdateStates(sActivationUp, raaW, raH, raaV0, raaH1d, raaH1s, rDropV, True)

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
                    # raaV2, junk  = self._UpdateStates(sActivationDn, raaW.T, raV, raaH1, rDropH)
                    self._UpdateStates(sActivationDn, raaW.T, raV, raaH1, raaV2, junk, rDropH)

                    # If we need to drop visible units...
                    if(rDropV>0):
                        
                        # Clear dropped states
                        raaV2.mult(baaV)

                    # Advance the markov chain V2->H3
                    # raaH3, junk  = self._UpdateStates(sActivationUp, raaW, raH, raaV2, rDropV)
                    self._UpdateStates(sActivationUp, raaW, raH, raaV2, raaH3, junk, rDropV)

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
                    
                    # Get a batch of inputs in raaV0
                    raaX.get_row_slice(iIndex, iIndex+self.iBatchSamples, target=raaV0)
                    
                    # Advance to the next minibatch
                    iIndex = iIndex + iBatch

                #
                raaXr = cudamat.empty((iSamples, iVs))

                # raaV2, junk  = self._UpdateStates(sActivationDn, raaW.T, raV, raaH1, 0)
                self._UpdateStates(sActivationUp, raaW, raH, raaX, raaY, junk, 0)
                
                # raaV2, junk  = self._UpdateStates(sActivationDn, raaW.T, raV, raaH1, 0)
                self._UpdateStates(sActivationDn, raaW.T, raV, raaY, raaXr, junk, 0)

                rTotalSe, rTotalE = self.GetErrors(raaX, raaXr, sActivationDn)
                    
                # Finish the rmse calculation
                rRmse = math.sqrt(rTotalSe/(raaX.shape[0]*raaX.shape[1]))
                
                # Finish rmse calculation
                rError = rTotalE/(raaX.shape[0]*raaX.shape[1])

                # Report training progress
                oOptions.fEvent(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError)
            
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
    
    # def UpdateStates(self, sType, raaW, raB, raaX, rDropout=0, bSample=False):
       
    #     baaY = [];

    #     # Compute the scale factor to compensate for dropout so that
    #     # average activations remain the same
    #     rScale = 1/(1-rDropout)

    #     print(raaX.shape)
        
    #     # Compute activations
    #     iRows = raaX.shape[0]

    #     raaA = numpy.dot(raaX*rScale, raaW)

    #     for iRow in range(iRows):
    #         raaA[iRow,:] = raaA[iRow,:] + raB
            
    #     # Depending on the activation type...
    #     if (sType=="Logistic"):

    #         # Compute the logistic def
    #         raaY = 1./(1+numpy.exp(-raaA))

    #     elif (sType=="Linear"):

    #         # Compute output layer states
    #         raaY = raaA

    #     elif (sType=="HyperbolicTangent"):

    #         # Compute output layer states
    #         raaY = numpy.tanh(raaA)
                                          
    #     # If stochastic binary states are required...
    #     if (bSample):

    #         # Depending on the activation type...
    #         if (sType=="Logistic"):

    #             # Sample output layer states
    #             baaY = raaY > numpy.random.random(raaY.shape)

    #         elif (sType=="Linear"):

    #             # Sample output layer states
    #             baaY = raaY + numpy.random.standard_normal(raaY.shape)

    #         elif (sType=="HyperbolicTangent"):

    #             # Sample output layer states
    #             baaY = raaY > 2*numpy.random.random(raaY.shape)-1

    #     return(raaY, baaY)

    def UpdateStates(self, sType, raaW, raB, raaX, rDropout=0, bSample=False):
       
        baaY = [];

        # Compute the scale factor to compensate for dropout so that
        # average activations remain the same
        rScale = 1/(1-rDropout)

        print(raaX.shape)
        
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

    def _UpdateStates(self, sType, _raaW, _raaB, _raaX, _raaY, _baaY, rDropout=0, bSample=False):

        #rDropout = 0;
       
        #_baaY = cudamat.CUDAMatrix(numpy.atleast_2d(0))

        # Compute the scale factor to compensate for dropout so that
        # average activations remain the same
        rScale = 1/(1-rDropout)
        
        # Compute activations
        #iRows = raaX.shape[0]
       # raaA = numpy.dot( numpy.concatenate( (raaX*rScale, numpy.ones(iRows).reshape(iRows,1) ), axis=1), raaW[:,:-1] )

        #_raaA = cudamat.CUDAMatrix(raaA)

        #_raaX = cudamat.empty((raaX.shape[0],raaX.shape[1]+1);
        #_raaX = cudamat.CUDAMatrix(raaX)
        #_raaX = _raaX.mult(rScale)
        #_raaW = cudamat.CUDAMatrix(raaW)

        #_raaB = cudamat.CUDAMatrix(numpy.atleast_2d(raB))
        _raaX.dot(_raaW,_raaY)
        _raaY = _raaY.mult(rScale)
        _raaY.add_row_vec(_raaB)

        # allocate outputs
        #_raaY = cudamat.empty(_raaA.shape)
            
        # Depending on the activation type...
        if (sType=="Logistic"):

            # Compute the logistic def
       #     raaY = 1./(1+numpy.exp(-raaA))
            _raaY.apply_sigmoid(_raaY)

        elif (sType=="Linear"):

            pass

            # Compute output layer states
       #     raaY = raaA
            #_raaY.assign(_raaY)

        elif (sType=="HyperbolicTangent"):

            # Compute output layer states
        #    raaY = numpy.tanh(raaA)
            _raaY.apply_tanh(_raaY)
                                          
        # If stochastic binary states are required...
        if (bSample):

            # _baaY = cudamat.empty(_raaY.shape)

            # Depending on the activation type...
            if (sType=="Logistic"):

                # Sample output layer states
        #        baaY = raaY > numpy.random.random(raaY.shape)
                _baaY.fill_with_rand()
                _baaY.less_than(_raaY)

            elif (sType=="Linear"):

                # Sample output layer states
        #        baaY = raaY + numpy.random.standard_normal(raaY.shape)
                _baaY.fill_with_randn()
                _baaY.add(_raaY)

            elif (sType=="HyperbolicTangent"):

                # Sample output layer states
        #        baaY = raaY > 2*numpy.random.random(raaY.shape)-1
                _baaY.fill_with_rand()
                _baaY.mult(2)
                _baaY.sub(1)
                _baaY.less_than(_raaY)

        # return(_raaY.asarray(), _baaY.asarray())
    
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
    
    def Autoencode(self, _raaX):
        
        # Maximum number of patterns to process at once
        iBatch = 1000
        
        # Measure the training data
        iSamples = _raaX.shape[0]
        
        raaX = cudamat.CUDAMatrix(_raaX)

        # Clear the index
        iIndex = 0
        
        # While training samples remain...
        while(iIndex<iSamples):

            # Compute an indexer
            # ia = range(iIndex, min(iIndex+iBatch,iSamples))

            raaB = cudamat.empty((iBatch,raaX.shape[1]))
            
            # Get a batch of inputs in raaV0
            raaX.get_row_slice(iIndex, iIndex+iBatch, target=raaB)

            junk = [];

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)-1):

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

                self._UpdateStates(sActivationUp, raaW, raH, raaB, raaY, junk)

                raaB = raaY

            # For each layer in the network...
            for iLayer in range(len(self.oaLayer)-2, -1, -1):

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

                self._UpdateStates(sActivationDn, raaW.T, raV, raaB, raaY, junk)

                raaB = raaY

                # Propagate states downward:
                #(raaB, junk) = self.UpdateStates(self.oaLayer[iLayer].sActivationDn, self.oaLayer[iLayer].raaW.T,self.oaLayer[iLayer].raV, raaB)
            
            # Save reconstruction states
            #raaX[ia,:] = raaB
            raaX.set_row_slice(iIndex, iIndex+iBatch, raaB)
            
            # Advance to the next batch
            iIndex = iIndex + iBatch

        _raaX = raaX.asarray()

        return(_raaX)    

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
        iBatch = 20000
        
        # Autoencode the specified samples to form reconstructions
        print('Autoencode')
        raaY = self.Autoencode(raaX)
        
        print('XY')
        raaX = cudamat.CUDAMatrix(raaX)
        raaY = cudamat.CUDAMatrix(raaY)

        # Measure the data
        iSamples = raaX.shape[0]

        print('GetErrors')
        rTotalSe, rTotalE = self.GetErrors(raaX, raaY, self.oaLayer[0].sActivationDn)
            
        # Finish the rmse calculation
        rRmse = math.sqrt(rTotalSe/(raaX.shape[0]*raaX.shape[1]))
        
        # Finish rmse calculation
        rError = rTotalE/(raaX.shape[0]*raaX.shape[1])
        
        # # Clear the index
        # iIndex = 0
        
        # # Clear the error accumulators
        # rTotalE  = 0
        # rTotalSe = 0
        
        # # While training samples remain...
        # while(iIndex<iSamples):
            
        #     # Compute an indexer
        #     ia = range(iIndex, min(iIndex+iBatch,iSamples))

        #     raaXs = raaX.get_row_slice(iIndex, min(iIndex+iBatch,iSamples))
        #     raaYs = raaY.get_row_slice(iIndex, min(iIndex+iBatch,iSamples))
            
        #     # Get errors for this batch
        #     (rSe, rE) = self.GetErrors(raaXs, raaYs, self.oaLayer[0].sActivationDn)
            
        #     # Accumulate the error totals
        #     rTotalSe = rTotalSe + rSe
        #     rTotalE = rTotalE + rE

        #     # Increment the index
        #     iIndex = iIndex + len(ia)

        # (rSe, rE) = self.GetErrors(raaXs, raaYs, self.oaLayer[0].sActivationDn)
        
        # # Average error over all samples
        # rError = rTotalE/raaX.size
        
        # # Root mean square error over all samples
        # rRmse  = math.sqrt(rTotalSe/raaX.size)
        
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
        rEps = 1e-20         
        
        raaError = cudamat.empty(raaX.shape)
        raaX.subtract(raaY, raaError)
        raaError.mult(raaError)
        raError = raaError.sum(axis=0)
        rError = raError.sum(axis=1)
        rSe = rError.asarray()[0,0]
        #print(rSe)

        # # Sum all squared errors
        # rSe = numpy.sum(numpy.square(raaError))

        # # Depending on the activation def type
        # if(sActivation=="Logistic"):

        #     # Compute the average cross entropy error
        #     rE = -numpy.sum(numpy.multiply(raaX,numpy.log(raaY+rEps)) + numpy.multiply(1-raaX,numpy.log(1-raaY+rEps)))

        # elif(sActivation=="Linear"):

        #     # Compute the squared error
        #     rE = rSe

        # elif(sActivation=="Softmax"):

        #     # Compute the average cross entropy error
        #     rE = -numpy.sum(numpy.multiply(raaX,numpy.log(raaY+rEps)))

        rE = 0
          
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

        def __init__(self, iSize, iEpochs, sActivationUp='Logistic'):
            self.iSize = iSize
            self.raaW = object
            self.sActivationUp = sActivationUp
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
            self.oaLayer = [LayerOptions(iEpochs), LayerOptions(iEpochs), LayerOptions(iEpochs), LayerOptions(iEpochs), LayerOptions(iEpochs)]  

    # Specify epochs
    iEpochs = 10

    # Read the MNIST dataset as a pandas.DataFrame
    df = pandas.read_pickle("../Datasets/MNIST/MNIST.pkl")

    # Retrieve the pixel columns and scale them from zero to one
    raaX = numpy.array(df.ix[:9999,0:783])/256.0

    # Create 784 x 1000 x 30 rbm layers
    oaLayers = [Layer(raaX.shape[1],iEpochs),Layer(1000,iEpochs),Layer(500,iEpochs),Layer(250,iEpochs),Layer(30,iEpochs,sActivationUp='Linear')]

    # Create training options
    oOptions = Options(iEpochs)

    # Create RbmStack
    oRbmStack = RbmStack(oaLayers)

    # Train using the specified options
    oRbmStack.TrainAutoencoder(raaX, oOptions)

    #print(o.oaLayer[1].raaW)

#import cProfile
#cProfile.run('Test()')
#Test()

