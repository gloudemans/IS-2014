import math
import numpy
import copy
try:
    import cudamat
    cudamat.init()
    cudamat.CUDAMatrix.init_random(seed = 42)
    bCudamatLoaded = True
except ImportError:
    bCudamatLoaded = False

## SequenceDecimatingNetwork
# This class implements a sequence decimating network. The network provides
# the following API methods:
#
# * SequenceDecimatingNetwork
# * ComputeOutputs - Compute the output patterns corresponding to a block of input patterns
# * ComputeGradient - Compute the gradient of error with respect to the network parameters
# * GetGradientVector - Get the gradient error with respect to all network parameters as a vector
# * GetWeightVector - Get all of the network parameters as a vector
# * SetWeightVector - Set all of the network parameters from a vector
# * Train - Train the network using the specified input and output patterns
#
# Pattern - a sequence of samples
# Sample - a set of features
# Feature - a parameter
#
# The network provides the following internal or test methods:
# * ComputeGradientNumerical - Compute the gradient numerically rather than analytically
# * TestGradient - Test the analytical gradient computation by comparison with the numerical gradient computation
# * TestTrain - Test training by training one network to mimic another randomly defined network

## Layer
# This contains the parameters that descibe a network layer.

class Layer:

	## Layer(self, iDecimation, raaW, raH)
	#
	# * iDecimation - decimation ratio for this layer
	# * raaW[iInputs, iOutputs] - specifies the connection weight matrix for this layer  
	# * raH[iOutputs] - specifies the bias vector for this layer

	def __init__(self, iDecimation, raaW, raH, raV=None):

		# Number of input samples per output summary			
		self.iDecimation = iDecimation

		# Connection weights from input to output
		self.raaW = numpy.copy(raaW)

		# Biases for each output
		self.raH  = numpy.copy(raH)

		# Biases for each input
		self.raV  = numpy.copy(raV)

	def __eq__(self, other):
		bEqual = self.iDecimation == other.iDecimation
		bEqual &= numpy.array_equal(self.raaW, other.raaW)
		bEqual &= numpy.array_equal(self.raH, other.raH)
		bEqual &= numpy.array_equal(self.raV, other.raV)

		bEqual &= self.raaW.dtype == other.raaW.dtype
		bEqual &= self.raH.dtype == other.raH.dtype
		bEqual &= self.raV.dtype == other.raV.dtype
		return bEqual

	def hash(self):
		rHash  = self.iDecimation
		rHash += numpy.sum(self.raaW)
		rHash += numpy.sum(self.raH)
		rHash += numpy.sum(self.raV)

		return rHash

## State
# State variables used during output computation and
# gradient computation for a layer.

class State:

	## State(self)

	def __init__(self):

		# Layer inputs [iSamples, iInputs]		
		self.raaX = numpy.zeros((0,0))

		# Layer derivatives [iSamples, iInputs]
		self.raaD = numpy.zeros((0,0))

		# Weight gradients [iInputs, iOutputs]
		self.raaWg = numpy.zeros((0,0))

		# Bias gradients [iOutpus]
		self.raaBg = numpy.zeros((0,0))

	def __eq__(self, other):
		bEqual =  numpy.array_equal(self.raaX, other.raaX)
		bEqual &= numpy.array_equal(self.raaD, other.raaD)
		bEqual &= numpy.array_equal(self.raaWg, other.raaWg)
		bEqual &= numpy.array_equal(self.raaBg, other.raaBg)

		bEqual &= self.raaX.dtype == other.raaX.dtype
		bEqual &= self.raaD.dtype == other.raaD.dtype		
		bEqual &= self.raaWg.dtype == other.raaWg.dtype
		bEqual &= self.raaBg.dtype == other.raaBg.dtype
		return bEqual

	def hash(self):

		rHash  = numpy.sum(self.raaX[:])
		rHash += numpy.sum(self.raaD[:])
		rHash += numpy.sum(self.raaWg[:])
		rHash += numpy.sum(self.raaWg[:])

		return rHash

class SequenceDecimatingNetwork:

	# Small offset constant to prevent log underflows when computing cross entropy 
	rEps = 1e-10

	# Set the batch size
	iBatch  = 100

	## SequenceDecimatingNetwork(self, oaLayers)
	# Construct a new SequenceDecimatingNetwork using the specified list
	# of network layers. The number of inputs for each layer must equal the
	# number of outputs for the prior layer multiplied by the decimation ratio
	# for the current layer. That is: 
	#   oaLayer[n].raaW.shape[0] = oaLayer[n-1].raaW.shape[1]*oaLayer[n].iDecimation
	# The sample decimation ratio for the network is the product of the decimation
	# ratios for its layers.
	#
	# * oaLayers - specifies the list connection layers

	def __init__(self, oaLayers, bUseGpu=True):

		# Set the GPU flag
		self.bUseGpu = bUseGpu and bCudamatLoaded

		# List of connection layers
		self.oaLayers = list(oaLayers)

		# Number of connection layers in the network
		self.iLayers = len(self.oaLayers)

		for oLayer in self.oaLayers:
			oLayer.raaW = numpy.copy(oLayer.raaW)
			oLayer.raH = numpy.copy(oLayer.raH)
			oLayer.raV = numpy.copy(oLayer.raV)	

		# List of states
		self.oaStates = []

		# Clear weight and bias counts
		self.iWeights = 0
		self.iBiases  = 0

		# For each layer...
		for iLayer in range(self.iLayers):

			# Count weights
			self.iWeights += self.oaLayers[iLayer].raaW.size

			# Count biases
			self.iBiases  += self.oaLayers[iLayer].raH.size

			# Create a state 
			self.oaStates.append(State())

		# Need an extra state for the network output
		self.oaStates.append(State())

		if(self.bUseGpu):

			# Create parameter vector
			self.raP = cudamat.empty((1,self.iWeights+self.iBiases))

			# For each layer...
			iBase = 0
			for iLayer in range(self.iLayers):

				# Count weights
				iChunk = self.oaLayers[iLayer].raaW.size

				raaW = cudamat.CUDAMatrix(self.oaLayers[iLayer].raaW.T)
				(x,y) = raaW.shape
				self.raP.set_col_slice(iBase,iBase+iChunk,raaW.reshape((1,iChunk)))
				self.oaLayers[iLayer].raaW = self.raP.get_col_slice(iBase,iBase+iChunk)
				self.oaLayers[iLayer].raaW.reshape((x,y))

				iBase += iChunk

			# For each layer...
			for iLayer in range(self.iLayers):

				# Count weights
				iChunk = self.oaLayers[iLayer].raH.size

				raH = cudamat.CUDAMatrix(numpy.atleast_2d(self.oaLayers[iLayer].raH).T)
				(x,y) = raH.shape
				self.raP.set_col_slice(iBase,iBase+iChunk,raH.reshape((1,iChunk)))
				self.oaLayers[iLayer].raH = self.raP.get_col_slice(iBase,iBase+iChunk)
				self.oaLayers[iLayer].raH.reshape((x,y))

				iBase += iChunk		

		else:

			self.raP  = numpy.empty((self.iWeights+self.iBiases))

			# For each layer...
			iBase = 0
			for iLayer in range(self.iLayers):

				# Count weights
				iChunk = self.oaLayers[iLayer].raaW.size

				(x,y) = self.oaLayers[iLayer].raaW.shape
				self.raP[iBase:iBase+iChunk] = self.oaLayers[iLayer].raaW.flatten()
				self.oaLayers[iLayer].raaW = self.raP[iBase:iBase+iChunk].reshape((x,y))

				iBase += iChunk

			# For each layer...
			for iLayer in range(self.iLayers):

				# Count weights
				iChunk = self.oaLayers[iLayer].raH.size

				self.raP[iBase:iBase+iChunk] = self.oaLayers[iLayer].raH.flatten()
				self.oaLayers[iLayer].raH = self.raP[iBase:iBase+iChunk]

				iBase += iChunk	

	def __eq__(self, other):
		bEqual  = self.bUseGpu  == other.bUseGpu
		bEqual &= self.iLayers  == other.iLayers
		bEqual &= self.iWeights == other.iWeights
		bEqual &= self.iBiases  == other.iBiases
		bEqual &= len(self.oaLayers) == len(other.oaLayers)
		bEqual &= len(self.oaStates) == len(other.oaStates)
		bEqual &= numpy.array_equal(self.raP, other.raP)

		for iLayer in range(len(self.oaLayers)):
			bEqual &= self.oaLayers[iLayer] == other.oaLayers[iLayer]

		for iState in range(len(self.oaStates)):
			bEqual &= self.oaStates[iState] == other.oaStates[iState]				

		return bEqual

	def hash(self):

		rHash  = self.bUseGpu
		rHash += self.iLayers
		rHash += self.iWeights
		rHash += self.iBiases
		rHash += len(self.oaLayers)
		rHash += len(self.oaStates)
		rHash += numpy.sum(self.raP)

		for iLayer in range(len(self.oaLayers)):
			rHash += self.oaLayers[iLayer].hash()

		for iState in range(len(self.oaStates)):
			rHash += self.oaStates[iState].hash()				

		return rHash

	## (raaaY) = (self, raaaX, bComputeDerivatives=False)
	# Compute network outputs from the specified network input, optionally
	# computing derivatives to use for subsequent gradient computation. The input
	# and output are three dimensional arrays with the following shape:
	#
	# (iPatterns, iSamples, iFeatures)
	#
	# * raaaX - specifies network inputs
	# * bComputeDerivatives - specifies derivative computation
	# * raaaY - returns outputs

	def ComputeOutputs(self, raaaX, bComputeDerivatives=False):

		# Measure the input
		(iPatterns, iInputLayerSamples, iFeatures) = raaaX.shape

		# Make 2d
		raaX = numpy.reshape(raaaX,(iPatterns*iInputLayerSamples,iFeatures))

		# If using gpu ...
		if(self.bUseGpu):

			(raaY, iDecimation) = self.ComputeOutputsGPU(cudamat.CUDAMatrix(raaX.T), bComputeDerivatives)

			raaaY = numpy.reshape(raaY.asarray().T,(iPatterns,iInputLayerSamples/iDecimation,-1))

		else:

			(raaY, iDecimation) = self.ComputeOutputsCPU(raaX, bComputeDerivatives)

			raaaY = numpy.reshape(raaY,(iPatterns,iInputLayerSamples/iDecimation,-1))

		return(raaaY)

	def ComputeOutputsCPU(self, raaX, bComputeDerivatives=False):

		# Save input samples
		self.oaStates[0].raaX  = raaX

		# Initialize overall decimation ratio
		iDecimation = 1

		# For each layer...
		for iLayer in range(self.iLayers):

			# Measure layer input
			(iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape

			# Aggregate features from multiple samples
			iFeatures *= self.oaLayers[iLayer].iDecimation

			# Update overall sample decimation ratio
			iDecimation *= self.oaLayers[iLayer].iDecimation

			# Decimate the layer
			self.oaStates[iLayer].raaX = numpy.reshape(self.oaStates[iLayer].raaX, (-1, iFeatures))

			# Compute activation function input
			self.oaStates[iLayer+1].raaX = numpy.dot(self.oaStates[iLayer].raaX, self.oaLayers[iLayer].raaW)

			# For each sample...
			for iSample in range(self.oaStates[iLayer+1].raaX.shape[0]):

				# Add bias
				self.oaStates[iLayer+1].raaX[iSample,:] += self.oaLayers[iLayer].raH

			# Compute logistic(x) activation
			self.oaStates[iLayer+1].raaX = 1./(1+numpy.exp(-self.oaStates[iLayer+1].raaX))

			# If this derivative is needed for backpropagation
			if(bComputeDerivatives and (iLayer<self.iLayers-1)):

				# Compute logistic'(x) activation derivative
				self.oaStates[iLayer+1].raaD = (1-self.oaStates[iLayer+1].raaX)*self.oaStates[iLayer+1].raaX

		# return(raaY, iDecimation)
		return(self.oaStates[self.iLayers].raaX, iDecimation)

	def ComputeInputs(self, raaaX):

		# Measure the input
		(iPatterns, iInputLayerSamples, iFeatures) = raaaX.shape

		# Make 2d
		raaX = numpy.reshape(raaaX,(iPatterns*iInputLayerSamples,iFeatures))

		# Save input samples
		raaX = self.oaStates[self.iLayers].raaX

		# Initialize overall decimation ratio
		iDecimation = 1

		# For each layer...
		for iLayer in reversed(range(self.iLayers)):

			print(iLayer, raaaX.shape, self.oaLayers[iLayer].raaW.T.shape, raaX.shape) 

			# Update overall sample decimation ratio
			iDecimation *= self.oaLayers[iLayer].iDecimation

			raaX = numpy.dot(raaX, self.oaLayers[iLayer].raaW.T)

			# For each sample...
			for iSample in range(raaX.shape[0]):

			 	# Add bias
				raaX[iSample,:] += self.oaLayers[iLayer].raV

			# Measure layer input
			(iSamples, iFeatures) = raaX.shape

			# Compute logistic(x) activation
			raaX = 1./(1+numpy.exp(-raaX))

			# Aggregate features from multiple samples
			iFeatures /= self.oaLayers[iLayer].iDecimation

			raaX = numpy.reshape(raaX, (-1, iFeatures))

		raaaX = numpy.reshape(raaX,(iPatterns,iInputLayerSamples*iDecimation,-1))

		# return(raaY, iDecimation)
		return(raaaX)

	def ComputeOutputsGPU(self, raaX, bComputeDerivatives):

		self.oaStates[0].raaX = raaX
	
		# Initialize overall decimation ratio
		iDecimation = 1

		# For each layer...
		for iLayer in range(self.iLayers):

			# Measure layer input
			# (iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape
			(iFeatures, iSamples) = self.oaStates[iLayer].raaX.shape

			# Aggregate features from multiple samples
			iFeatures *= self.oaLayers[iLayer].iDecimation

			# Update overall sample decimation ratio
			iDecimation *= self.oaLayers[iLayer].iDecimation

			# Decimate the layer
			# self.oaStates[iLayer].raaX = numpy.reshape(self.oaStates[iLayer].raaX, (-1, iFeatures))
			iSize = numpy.prod(self.oaStates[iLayer].raaX.shape)
			self.oaStates[iLayer].raaX = self.oaStates[iLayer].raaX.reshape((iFeatures,iSize//iFeatures))
	
			# Compute activation function input
			# self.oaStates[iLayer+1].raaX = numpy.dot(self.oaStates[iLayer].raaX, self.oaLayers[iLayer].raaW)
			self.oaStates[iLayer+1].raaX = cudamat.dot(self.oaLayers[iLayer].raaW, self.oaStates[iLayer].raaX)

			# For each sample...add bias
			# 	self.oaStates[iLayer+1].raaX[iSample,:] += self.oaLayers[iLayer].raH
			self.oaStates[iLayer+1].raaX.add_col_vec(self.oaLayers[iLayer].raH)

			# Compute logistic(x) activation
			# self.oaStates[iLayer+1].raaX = 1./(1+numpy.exp(-self.oaStates[iLayer+1].raaX))
			self.oaStates[iLayer+1].raaX.apply_sigmoid()

			# If this derivative is needed for backpropagation
			if(bComputeDerivatives and (iLayer<self.iLayers-1)):

				# Compute logistic'(x) activation derivitive
				# self.oaStates[iLayer+1].raaD = (1-self.oaStates[iLayer+1].raaX)*self.oaStates[iLayer+1].raaX
				self.oaStates[iLayer+1].raaD = cudamat.empty(self.oaStates[iLayer+1].raaX.shape)
				self.oaStates[iLayer+1].raaD.assign(1)
				self.oaStates[iLayer+1].raaD.subtract(self.oaStates[iLayer+1].raaX)
				self.oaStates[iLayer+1].raaD.mult(self.oaStates[iLayer+1].raaX)

		# return(raaY, iDecimation)
		return(self.oaStates[self.iLayers].raaX, iDecimation)

	## (raG) = ComputeGradient(self, raaaX, raaaT)
	# Compute the gradient of error with respect to all learnable network parameters.
	#
	# (iPatterns, iSamples, iFeatures)
	#
	# * raaaX - specifies network inputs
	# * raaaT - specifies the desired network outputs
	# * raG - returns gradient of error with respect to network parameters as a vector
	# * rError - returns cross entropy
	# * rRmse - returns root mean square error

	def ComputeGradient(self, raaaX, raaaT, bFinalLayer=False):


		# Measure the input
		(iPatternsX, iPatternSamplesX, iPatternFeaturesX) = raaaX.shape
		(iPatternsT, iPatternSamplesT, iPatternFeaturesT) = raaaT.shape

		# Make 2d
		raaX = numpy.reshape(raaaX,(iPatternsX*iPatternSamplesX,iPatternFeaturesX))
		raaT = numpy.reshape(raaaT,(iPatternsT*iPatternSamplesT,iPatternFeaturesT))

		# Save input samples
		if(self.bUseGpu):
			raaX = cudamat.CUDAMatrix(raaX.T)
			raaT = cudamat.CUDAMatrix(raaT.T)

			(raaY, iDecimation) = self.ComputeOutputsGPU(raaX, bComputeDerivatives=True)
			raaE = raaY.copy()
			raaE.subtract(raaT)
			raG = self.ComputeGradientGPU(raaE, bFinalLayer)

		else:

			(raaY, iDecimation) = self.ComputeOutputsCPU(raaX, bComputeDerivatives=True)
			raaE = raaY-raaT
			raG = self.ComputeGradientCPU(raaE, bFinalLayer)

		return(raG)

	def ComputeGradientCPU(self, raaE, bFinalLayer=False):

		# For each layer...
		for iLayer in range(self.iLayers-1,-1,-1):

			# Measure the layer input
			(iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape	

			# Compute the gradient of error with respect to weight
			self.oaStates[iLayer].raaWg = numpy.dot(self.oaStates[iLayer].raaX.T, raaE)

			# Compute gradient of error with respect to bias
			self.oaStates[iLayer].raBg = numpy.sum(raaE,0)

			# If error is needed for next layer...
			if(iLayer>0):

				# Backpropagate the error
				raaE = numpy.dot(raaE,self.oaLayers[iLayer].raaW.T)

				# Compute the sample count for prior layer
				iSamples = raaE.shape[0]*self.oaLayers[iLayer].iDecimation

				# Undecimate error
				raaE = numpy.reshape(raaE,(iSamples,-1))

				# Compute deferred hadamard product with derivative so shapes match
				raaE = raaE*self.oaStates[iLayer].raaD		

		# Get the serialized gradient vector
		raG = self.GetGradientVector(bFinalLayer)

		# Return gradient and error metrics
		return(raG)

	def ComputeGradientGPU(self, raaE, bFinalLayer=False):

		# For each layer...
		for iLayer in range(self.iLayers-1,-1,-1):

			# Measure the layer input
			# (iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape
			(iFeatures, iSamples) = self.oaStates[iLayer].raaX.shape			

			# Compute the gradient of error with respect to weight
			# self.oaStates[iLayer].raaWg = numpy.dot(self.oaStates[iLayer].raaX.T, raaE)
			self.oaStates[iLayer].raaWg = cudamat.dot(self.oaStates[iLayer].raaX, raaE.T)

			# Compute gradient of error with respect to bias
			# self.oaStates[iLayer].raBg = numpy.sum(raaE,0)
			self.oaStates[iLayer].raBg = raaE.sum(1)

			# If error is needed for next layer...
			if(iLayer>0):

				# Backpropagate the error
				# raaE = numpy.dot(raaE,self.oaLayers[iLayer].raaW.T)
				raaE = cudamat.dot(self.oaLayers[iLayer].raaW.T, raaE)

				# Compute the sample count for prior layer
				# iSamples = raaE.shape[0]*self.oaLayers[iLayer].iDecimation
				iSamples = raaE.shape[1]*self.oaLayers[iLayer].iDecimation

				# Undecimate error
				# raaE = numpy.reshape(raaE,(iSamples,-1))
				iSize = numpy.prod(raaE.shape)
				iN = iSize//iSamples
				raaE.reshape((iN,iSamples))

				# Compute deferred hadamard product with derivative so shapes match
				# raaE = raaE*self.oaStates[iLayer].raaD
				raaE.mult(self.oaStates[iLayer].raaD)				

		# Get the serialized gradient vector
		raG = self.GetGradientVector(bFinalLayer)

		# Return gradient and error metrics
		# return((raG, rError, rRmse))
		return(raG)

	## Train(self, raaaX, raaaT, iPatterns, rRate, rMomentum, fProgress=None)
	# Train the network using the specified input patterns, desired output patterns,
	# rate and momentum, and report progress using the specified callback. Training
	# uses stochastic gradient descent algorithm with momentum. Network inputs and outputs
	# are three dimensional arrays organized as follows:
	#
	# (iPatterns, iSamples, iFeatures)
	#
	# * raaaX - specifies network inputs
	# * raaaT - specifies the desired network outputs
	# * iPatterns - specifies the number of training patterns to process
	# * rRate - specifies the learning rate
	# * rMomentum - specifies the momentum to apply
	# * fProgress - specifies the progress callback

	def Train(self, raaaX, raaaT, iPatterns, rRate, rMomentum, bFinalLayer=False, fProgress=None):

		# Clear the batch start index
		i0 = 0

		# Retrieve the weight vector
		raW = self.GetWeightVector()

		# Create momentum array
		raDelta = numpy.zeros(raW.shape)

		# Clear the pattern counter
		iPattern = 0

		# Measure the input
		(iPatternsX, iPatternSamplesX, iPatternFeaturesX) = raaaX.shape
		(iPatternsT, iPatternSamplesT, iPatternFeaturesT) = raaaT.shape

		# Make 2d
		raaX = numpy.reshape(raaaX,(iPatternsX*iPatternSamplesX,iPatternFeaturesX))
		raaT = numpy.reshape(raaaT,(iPatternsT*iPatternSamplesT,iPatternFeaturesT))

		# Save input samples
		if(self.bUseGpu):
			raaX = cudamat.CUDAMatrix(raaX.T)
			raaT = cudamat.CUDAMatrix(raaT.T)

		# While there are more patterns to process...
		while(iPattern<iPatterns):

			# Compute a batch stop index that won't exceed the pattern count or the number of patterns in the input 
			i1 = min(i0 + min(self.iBatch,iPatterns-iPattern), iPatternsX)

			if(self.bUseGpu):

				# Slice the batch
				_raaXs = raaX.get_col_slice(i0*iPatternSamplesX,i1*iPatternSamplesX)
				_raaTs = raaT.get_col_slice(i0*iPatternSamplesT,i1*iPatternSamplesT)
				(_raaYs, iDecimation) = self.ComputeOutputsGPU(_raaXs, bComputeDerivatives=True)
				raaE = _raaYs.copy()
				raaE.subtract(_raaTs)

				# Compute the gradient
				#(raG, rError, rRmse) = self.ComputeGradient(raaaXs, raaaTs)
				raG = self.ComputeGradientGPU(raaE, bFinalLayer)

				# # Flatten the network outputs and targets to simplify error metrics
				raY = _raaYs.asarray().flatten()
				raT = _raaTs.asarray().flatten()
			else:

				_raaXs = raaX[i0*iPatternSamplesX:i1*iPatternSamplesX,:]
				_raaTs = raaT[i0*iPatternSamplesT:i1*iPatternSamplesT,:]

				(_raaYs, iDecimation) = self.ComputeOutputsCPU(_raaXs, bComputeDerivatives=True)
				raaE = _raaYs-_raaTs
				raG = self.ComputeGradientCPU(raaE, bFinalLayer)

				raY = _raaYs.flatten()
				raT = _raaTs.flatten()

			# Increment the number of patterns procressed
			iPattern += i1-i0

			#print("raG ",numpy.sum(raG))

			# Advance the batch start index
			i0 = i1

			# Compute cross entropy error
			rError = -numpy.mean(raT*numpy.log(raY+self.rEps) + (1-raT)*numpy.log(1-raY+self.rEps))

			# Compute root mean square error
			rRmse  = numpy.sqrt(numpy.mean((raT-raY)**2))

			# If progress callback specified...
			if(fProgress):

				# Call it
				fProgress(iPattern, rError, rRmse)

			# Update the weight with momentum
			raDelta[0:self.iWeights] = raDelta[0:self.iWeights]*rMomentum + raG[0:self.iWeights]*rRate

			# Update the biases with no momentum
			raDelta[self.iWeights:] = raG[self.iWeights:]*rRate

			#print("raG", numpy.sum(raG))
			#print("raW", numpy.sum(raW))

			# Update the local weights
			#raW = raW - raDelta
			raW[0] += 1

			#print("raW", numpy.sum(raW))
			#print("raDelta", numpy.sum(raDelta))

			# Insert updated weights into the network
			self.SetWeightVector(raW)

			for k in range(len(self.oaLayers)):
				print(k,numpy.sum(self.oaLayers[k].raaW))
			#print("raW", numpy.sum(raW))			

			#print("raP", numpy.sum(self.raP))

			# If we've reached the end of the input array...
			if(i0==iPatternsX):

				# Wrap to the beginning
				i0 = 0

			print(iPattern, self.hash())

	## (raG) = GetGradientVector(self)
	# Get the gradient of error with respect to all learnable network parameters as a vector.
	#
	# * raG - returns error gradient

	def GetGradientVector(self, bFinalLayer=False):

		# Start with an empty array
		raG = numpy.array([])

		# For each layer...
		for iLayer in range(self.iLayers):

			if(self.bUseGpu):

				# Get flattened weight gradients
				raChunk = self.oaStates[iLayer].raaWg.asarray().ravel()

			else:

				# Get the flattened weight gradients
				raChunk = self.oaStates[iLayer].raaWg.ravel()

			# If training the final layer only and this isn't the final layer...
			if(bFinalLayer and (iLayer<self.iLayers-1)):

				# Clear the gradient
				raChunk[:] = 0

			# Concatenate this chunk with the gradient vector
			raG = numpy.concatenate((raG, raChunk))

		# For each layer...
		for iLayer in range(self.iLayers):

			if(self.bUseGpu):

				# Get the flattened bias gradients
				raChunk = self.oaStates[iLayer].raBg.asarray().ravel()

			else:

				# Get the flattened bias gradients
				raChunk = self.oaStates[iLayer].raBg.ravel()

			# If training the final layer only and this isn't the final layer...
			if(bFinalLayer and (iLayer<self.iLayers-1)):

				# Clear the gradient
				raChunk[:] = 0

			# Concatenate this chunk with the gradient vector
			raG = numpy.concatenate((raG, raChunk))

		# Return the error gradient vector
		return(raG)

	## (raW) = GetWeightVector
	# Get all learnable network parameters as a vector.
	#
	# * raW - returns parameter vector

	def GetWeightVector(self):

		if(self.bUseGpu):

			return(self.raP.asarray().squeeze())

		else:

			return(self.raP)

	## SetWeightVector(self, raW)
	# Set all learnable network parameters from a parameter vector.
	#		return(self.raP.asarray().squeeze().copy())

	# * raW - specifies the parameter vector

	def SetWeightVector(self, raW):

		if(self.bUseGpu):

			self.raP.assign(cudamat.CUDAMatrix(numpy.atleast_2d(raW)))

		else:

			self.raP[:] = raW

	## (raG) = ComputeGradientNumerical(self, raaX, raaT, rDelta=1e-6)
	# Numerically compute the gradient of error with respect to all learnable 
	# network parameters.
	#
	# (iPatterns, iSamples, iFeatures)
	#
	# * raaaX - specifies network inputs
	# * raaaT - specifies the desired network outputs
	# * rDelta - weight delta for slope computation
	# * raG - returns gradient of error with respect to network parameters as a vector

	def ComputeGradientNumerical(self, raaaX, raaaT, rDelta=1e-3):

		# Compute flattened outputs
		raY = self.ComputeOutputs(raaaX).flatten()

		# Compute flattened targets
		raT = raaaT.flatten()

		# Compute cross entropy
		rError = -numpy.sum(raT*numpy.log(raY+self.rEps) + (1-raT)*numpy.log(1-raY+self.rEps))

		# Get the weight vector
		raW = self.GetWeightVector().copy()

		# Make gradient vector in same shape
		raG = numpy.zeros(raW.shape)

		# Copy the weight vector
		raWd = numpy.copy(raW)

		# For each parameter...
		for k in range(len(raW)):

			# Adjust the parameter by delta
			raWd[k] += rDelta

			# Update the network with with adjusted parameter
			self.SetWeightVector(raWd)

			# Compute flattened network outputs with adjusted parameter
			raY0 = self.ComputeOutputs(raaaX).flatten()

			# Restore original weight
			raWd[k] = raW[k]

			# Compute cross entropy with with adjusted parameter
			rError0 = -numpy.sum(raT*numpy.log(raY0+self.rEps) + (1-raT)*numpy.log(1-raY0+self.rEps))

			# Approximate derivative of error with respect to weight numerically
			raG[k] = (rError0-rError)/rDelta

		# Restore the original network weights
		self.SetWeightVector(raW)

		# Force comprehensive state update
		self.ComputeOutputs(raaaX, True)

		# Return the gradient
		return(raG)

## TestGradient()
# Configure a small random network. Compute the error gradient using backpropagation
# and compute it numerically. Compute and print an error measure between the two methods.

def TestGradient():

	# Specify number of test patterns
	iPatterns = 100;

	# Specify number of samples per pattern
	iPatternSamples = 6;

	# Specify scale of random inputs and random network parameters
	rScale = 1;

	# Create random input vectors
	raaaX = numpy.random.rand(iPatterns,iPatternSamples,3)

	# Create random target vectors
	raaaT = numpy.random.randn(iPatterns,1,2)*rScale

	# Create layer 0 with random weights and biases
	oL0 = Layer(3, numpy.random.randn(9,4)*rScale, numpy.random.randn(4)*rScale)

	# Create layer 1 with random weights and biases
	oL1 = Layer(2, numpy.random.randn(8,2)*rScale, numpy.random.randn(2)*rScale)

	# Create object
	o = SequenceDecimatingNetwork([oL0,oL1])

	# Compute gradient using backpropagation
	raG = o.ComputeGradient(raaaX, raaaT)

	# Compute gradient numerically
	raGn = o.ComputeGradientNumerical(raaaX, raaaT)

	# Compute normalized error measure
	rError = 2*sum((raG-raGn)**2)/sum(0.5*(raG**2+raGn**2))

	# Report error measure
	print("Gradient Test: rError={:f}".format(rError))

## TestTrain()
# Configure two small random networks withthe same geometry but different weights and biases.
# Train one network to model the other while displaying error measure.

def TestTrain():

	# Specify the number of training patterns to generate
	iPatterns = 1000

	# Integer value to change network scale
	iMagnify = 100

	# Specify the scale of random initialization parameters
	rScale = 0.01

	# Create layer 0 with random weights and biases
	oL0 = Layer(3, numpy.random.randn(9*iMagnify,4*iMagnify)*rScale, numpy.random.randn(4*iMagnify)*rScale)

	# Create layer 1 with random weights and biases
	oL1 = Layer(2, numpy.random.randn(8*iMagnify,2*iMagnify)*rScale, numpy.random.randn(2*iMagnify)*rScale)

	# Create object
	o0 = SequenceDecimatingNetwork([oL0, oL1])

	# Create layer 0 with random weights and biases
	oL0 = Layer(3, numpy.random.randn(9*iMagnify,4*iMagnify)*rScale, numpy.random.randn(4*iMagnify)*rScale)

	# Create layer 1 with random weights and biases
	oL1 = Layer(2, numpy.random.randn(8*iMagnify,2*iMagnify)*rScale, numpy.random.randn(2*iMagnify)*rScale)

	# Create object
	o1 = SequenceDecimatingNetwork([oL0, oL1])

	# Specify samples in input pattern
	iPatternSamples = 6

	# Create random input vectors
	raaaX = numpy.random.rand(iPatterns,iPatternSamples,3*iMagnify)

	# Create training vectors from network zero
	raaaT = o0.ComputeOutputs(raaaX)

	# Train network 1 to model network zero
	o1.Train(raaaX, raaaT,  1000, 0.01, 0.5, False, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))
	o1.Train(raaaX, raaaT, 10000, 0.01, 0.9, False, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))

#TestTrain()