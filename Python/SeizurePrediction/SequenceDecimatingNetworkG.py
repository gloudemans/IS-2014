import math
import numpy
import cudamat

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

class SequenceDecimatingNetwork:

	# Small offset constant to prevent log underflows when computing cross entropy 
	rEps = 1e-10

	# Set the batch size
	iBatch  = 100

	## Layer
	# This contains the parameters that descibe a network layer.

	class Layer:

		## Layer(self, iDecimation, raaW, raB)
		#
		# * iDecimation - decimation ratio for this layer
		# * raaW[iInputs, iOutputs] - specifies the connection weight matrix for this layer  
		# * raB[iOutputs] - specifies the bias vector for this layer

		def __init__(self, iDecimation, raaW, raB):

			# Number of input samples per output summary			
			self.iDecimation = iDecimation

			# Connection weights from input to output
			# self.raaW = raaW			
			self._raaW = cudamat.CUDAMatrix(raaW.T)

			# Biases for each output
			# self.raB  = raB
			self._raB  = cudamat.CUDAMatrix(numpy.atleast_2d(raB).T)

	## State
	# State variables used during output computation and
	# gradient computation for a layer.

	class State:

		## State(self)

		def __init__(self):

			# Layer inputs [iSamples, iInputs]
			# self.raaX = []			
			self._raaX = []

			# Layer derivatives [iSamples, iInputs]
			# self.raaD = []
			self._raaD = []

			# Weight gradients [iInputs, iOutputs]
			# self.raaWg = []
			self._raaWg = []

			# Bias gradients [iOutpus]
			# self.raaBg = []
			self._raaBg = []	

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

	def __init__(self, oaLayers):

		# Number of connection layers in the network
		self.iLayers = len(oaLayers)

		# List of connection layers
		self.oaLayers = oaLayers

		# List of states
		self.oaStates = []

		# Clear weight and bias counts
		self.iWeights = 0
		self.iBiases  = 0

		# For each layer...
		for iLayer in range(self.iLayers):

			# Count weights
			# self.iWeights += self.oaLayers[iLayer].raaW.size
			self.iWeights += numpy.prod(self.oaLayers[iLayer]._raaW.shape)

			# Count biases
			# self.iBiases  += self.oaLayers[iLayer].raB.size
			self.iBiases  += numpy.prod(self.oaLayers[iLayer]._raB.shape)

			# Create a state 
			self.oaStates.append(SequenceDecimatingNetwork.State())

		# Need an extra state for the network output
		self.oaStates.append(SequenceDecimatingNetwork.State())

		#self.raP  = numpy.empty((self.iWeights+self.iBiases))
		self._raP = cudamat.empty((1,self.iWeights+self.iBiases))

		# For each layer...
		iBase = 0
		for iLayer in range(self.iLayers):

			# Count weights
			# iChunk = self.oaLayers[iLayer].raaW.size

			# (x,y) = self.oaLayers[iLayer].raaW.shape
			# self.raP[iBase:iBase+iChunk] = self.oaLayers[iLayer].raaW.flatten()
			# self.oaLayers[iLayer].raaW = self.raP[iBase:iBase+iChunk].reshape((x,y))

			iChunk = numpy.prod(self.oaLayers[iLayer]._raaW.shape)

			(x,y) = self.oaLayers[iLayer]._raaW.shape
			self._raP.set_col_slice(iBase,iBase+iChunk,self.oaLayers[iLayer]._raaW.reshape((1,iChunk)))
			self.oaLayers[iLayer]._raaW = self._raP.get_col_slice(iBase,iBase+iChunk)
			self.oaLayers[iLayer]._raaW.reshape((x,y))

			iBase += iChunk

		# For each layer...
		for iLayer in range(self.iLayers):

			# Count weights
			# iChunk = self.oaLayers[iLayer].raB.size

			# self.raP[iBase:iBase+iChunk] = self.oaLayers[iLayer].raB.flatten()
			# self.oaLayers[iLayer].raB = self.raP[iBase:iBase+iChunk]

			iChunk = numpy.prod(self.oaLayers[iLayer]._raB.shape)

			(x,y) = self.oaLayers[iLayer]._raB.shape
			self._raP.set_col_slice(iBase,iBase+iChunk,self.oaLayers[iLayer]._raB.reshape((1,iChunk)))
			self.oaLayers[iLayer]._raB = self._raP.get_col_slice(iBase,iBase+iChunk)
			self.oaLayers[iLayer]._raB.reshape((x,y))

			iBase += iChunk		

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

		# Save input samples
		_raaX = cudamat.CUDAMatrix(raaX.T)

		(_raaY, iDecimation) = self.ComputeOutputsCore(_raaX, bComputeDerivatives)

		raaaY = numpy.reshape(_raaY.asarray().T,(iPatterns,iInputLayerSamples/iDecimation,-1))

		return(raaaY)

	def ComputeOutputsCore(self, _raaX, bComputeDerivatives=False):

	# 	# Measure the input
	# 	(iPatterns, iInputLayerSamples, iFeatures) = raaaX.shape

	# 	# Make 2d
	# 	raaX = numpy.reshape(raaaX,(iPatterns*iInputLayerSamples,iFeatures))

	# 	# Save input samples
	# 	# self.oaStates[0].raaX  = raaX
	# 	self.oaStates[0]._raaX = cudamat.CUDAMatrix(raaX.T)

		self.oaStates[0]._raaX = _raaX
	
		# Initialize overall decimation ratio
		iDecimation = 1

		# For each layer...
		for iLayer in range(self.iLayers):

			# Measure layer input
			# (iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape
			(iFeatures, iSamples) = self.oaStates[iLayer]._raaX.shape

			# Aggregate features from multiple samples
			iFeatures *= self.oaLayers[iLayer].iDecimation

			# Update overall sample decimation ratio
			iDecimation *= self.oaLayers[iLayer].iDecimation

			# Decimate the layer
			# self.oaStates[iLayer].raaX = numpy.reshape(self.oaStates[iLayer].raaX, (-1, iFeatures))

			iSize = numpy.prod(self.oaStates[iLayer]._raaX.shape)
			self.oaStates[iLayer]._raaX = self.oaStates[iLayer]._raaX.reshape((iFeatures,iSize//iFeatures))
	
			# Compute activation function input
			# self.oaStates[iLayer+1].raaX = numpy.dot(self.oaStates[iLayer].raaX, self.oaLayers[iLayer].raaW)
			self.oaStates[iLayer+1]._raaX = cudamat.dot(self.oaLayers[iLayer]._raaW, self.oaStates[iLayer]._raaX)

			# For each sample...
			# for iSample in range(self.oaStates[iLayer+1].raaX.shape[0]):

			# 	# Add bias
			# 	self.oaStates[iLayer+1].raaX[iSample,:] += self.oaLayers[iLayer].raB
			self.oaStates[iLayer+1]._raaX.add_col_vec(self.oaLayers[iLayer]._raB)

			# Compute logistic(x) activation
			# self.oaStates[iLayer+1].raaX = 1./(1+numpy.exp(-self.oaStates[iLayer+1].raaX))
			self.oaStates[iLayer+1]._raaX.apply_sigmoid()

			# If this derivative is needed for backpropagation
			if(bComputeDerivatives and (iLayer<self.iLayers-1)):

				# Compute logistic'(x) activation derivitive
				# self.oaStates[iLayer+1].raaD = (1-self.oaStates[iLayer+1].raaX)*self.oaStates[iLayer+1].raaX
				self.oaStates[iLayer+1]._raaD = cudamat.empty(self.oaStates[iLayer+1]._raaX.shape)
				self.oaStates[iLayer+1]._raaD.assign(1)
				self.oaStates[iLayer+1]._raaD.subtract(self.oaStates[iLayer+1]._raaX)
				self.oaStates[iLayer+1]._raaD.mult(self.oaStates[iLayer+1]._raaX)

		# Reshape as patterns, samples, features
		# raaaY = numpy.reshape(numpy.copy(self.oaStates[self.iLayers].raaX),(iPatterns,iInputLayerSamples/iDecimation,-1))
		#raaaY = numpy.reshape(self.oaStates[iLayer+1]._raaX.transpose().asarray(),(iPatterns,iInputLayerSamples/iDecimation,-1))

		#return(raaaY)
		return(self.oaStates[self.iLayers]._raaX, iDecimation)

	## (raG, rError, rRmse) = ComputeGradient(self, raaaX, raaaT)
	# Compute the gradient of error with respect to all learnable network parameters.
	#
	# (iPatterns, iSamples, iFeatures)
	#
	# * raaaX - specifies network inputs
	# * raaaT - specifies the desired network outputs
	# * raG - returns gradient of error with respect to network parameters as a vector
	# * rError - returns cross entropy
	# * rRmse - returns root mean square error

	# def ComputeGradient(self, raaaX, raaaT):	
	def ComputeGradient(self, _raaE):

		# # Compute the network outputs while saving derivatives
		# raaaY = self.ComputeOutputs(raaaX, bComputeDerivatives=True)

		# # Flatten the network outputs and targets to simplify error metrics
		# raY = raaaY.flatten()
		# raT = raaaT.flatten()

		# # Compute cross entropy error
		# rError = -numpy.mean(raT*numpy.log(raY+self.rEps) + (1-raT)*numpy.log(1-raY+self.rEps))

		# # Compute root mean square error
		# rRmse  = numpy.sqrt(numpy.mean((raT-raY)**2))

		# # Compute output layer error
		# raaE = raaaY - raaaT

		# # Coerce the shape
		# # raaE.shape = self.oaStates[self.iLayers].raaX.shape
		# raaE.shape = self.oaStates[self.iLayers]._raaX.shape[1],self.oaStates[self.iLayers]._raaX.shape[0]
		# _raaE = cudamat.CUDAMatrix(raaE.T)

		# For each layer...
		for iLayer in range(self.iLayers-1,-1,-1):

			# Measure the layer input
			# (iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape
			(iFeatures, iSamples) = self.oaStates[iLayer]._raaX.shape			

			# Compute the gradient of error with respect to weight
			# self.oaStates[iLayer].raaWg = numpy.dot(self.oaStates[iLayer].raaX.T, raaE)
			self.oaStates[iLayer]._raaWg = cudamat.dot(self.oaStates[iLayer]._raaX, _raaE.T)

			# Compute gradient of error with respect to bias
			# self.oaStates[iLayer].raBg = numpy.sum(raaE,0)
			self.oaStates[iLayer]._raBg = _raaE.sum(1)

			# If error is needed for next layer...
			if(iLayer>0):

				# Backpropagate the error
				# raaE = numpy.dot(raaE,self.oaLayers[iLayer].raaW.T)
				_raaE = cudamat.dot(self.oaLayers[iLayer]._raaW.T, _raaE)

				# Compute the sample count for prior layer
				# iSamples = raaE.shape[0]*self.oaLayers[iLayer].iDecimation
				iSamples = _raaE.shape[1]*self.oaLayers[iLayer].iDecimation

				# Undecimate error
				# raaE = numpy.reshape(raaE,(iSamples,-1))
				iSize = numpy.prod(_raaE.shape)
				iN = iSize//iSamples
				_raaE.reshape((iN,iSamples))

				# Compute deferred hadamard product with derivative so shapes match
				# raaE = raaE*self.oaStates[iLayer].raaD
				_raaE.mult(self.oaStates[iLayer]._raaD)				

		# Get the serialized gradient vector
		raG = self.GetGradientVector()

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

	def Train(self, raaaX, raaaT, iPatterns, rRate, rMomentum, fProgress=None):

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
		_raaX = cudamat.CUDAMatrix(raaX.T)
		_raaT = cudamat.CUDAMatrix(raaT.T)

		# While there are more patterns to process...
		while(iPattern<iPatterns):

			# Compute a batch stop index that won't exceed the pattern count or the number of patterns in the input 
			i1 = min(i0 + min(self.iBatch,iPatterns-iPattern), iPatternsX)

			# Slice the batch
			_raaXs = _raaX.get_col_slice(i0*iPatternSamplesX,i1*iPatternSamplesX) # raaaX[i0:i1,:,:]
			_raaTs = _raaT.get_col_slice(i0*iPatternSamplesT,i1*iPatternSamplesT) # raaaT[i0:i1,:,:]

			# Measure the input
			#(iPatternsS, iInputLayerSamples, iFeatures) = raaaXs.shape

			# Make 2d
			#raaX = numpy.reshape(raaaXs,(iPatternsS*iInputLayerSamples,iFeatures))

			#reshape(raaaXs,(iPatternsS*iInputLayerSamples,iFeatures))

			# Save input samples
			#_raaX = cudamat.CUDAMatrix(raaX.T)

			# Compute outputs and keep deivatives
			(_raaYs, iDecimation) = self.ComputeOutputsCore(_raaXs, bComputeDerivatives=True)

			#print(_raaTs.shape,iPatternSamplesX,iPatternFeaturesX)
			#print(_raaYs.shape,iPatternSamplesT,iPatternFeaturesT)

			#raaaYs = numpy.reshape(_raaY.asarray().T,(iPatternsS,iInputLayerSamples/iDecimation,-1))

			# Increment the number of patterns procressed
			iPattern += i1-i0

			# Advance the batch start index
			i0 = i1

			# FFF

			# # Compute the network outputs while saving derivatives
			# raaaY = self.ComputeOutputs(raaaX, bComputeDerivatives=True)

			# # Flatten the network outputs and targets to simplify error metrics
			raY = _raaYs.asarray().flatten()
			raT = _raaTs.asarray().flatten()

			# # Compute cross entropy error
			rError = -numpy.mean(raT*numpy.log(raY+self.rEps) + (1-raT)*numpy.log(1-raY+self.rEps))

			# # Compute root mean square error
			rRmse  = numpy.sqrt(numpy.mean((raT-raY)**2))

			# # Compute output layer error
			#raaE = raaaYs - raaaTs

			# # Coerce the shape
			# # raaE.shape = self.oaStates[self.iLayers].raaX.shape
			#raaE.shape = self.oaStates[self.iLayers]._raaX.shape[1],self.oaStates[self.iLayers]._raaX.shape[0]
			#_raaE = cudamat.CUDAMatrix(raaE.T)

			_raaE = _raaYs.copy()
			_raaE.subtract(_raaTs)

			# FFF
			#rError=0
			#rRmse = 0

			# Compute the gradient
			#(raG, rError, rRmse) = self.ComputeGradient(raaaXs, raaaTs)
			raG = self.ComputeGradient(_raaE)

			# If progress callback specified...
			if(fProgress):

				# Call it
				fProgress(iPattern, rError, rRmse)

			# Update the weight with momentum
			raDelta[0:self.iWeights] = raDelta[0:self.iWeights]*rMomentum + raG[0:self.iWeights]*rRate

			# Update the biases with no momentum
			raDelta[self.iWeights:] = raG[self.iWeights:]*rRate

			# Update the local weights
			raW = raW - raDelta;

			# Insert updated weights into the network
			self.SetWeightVector(raW)

			# If we've reached the end of the input array...
			if(i0==raaaX.shape[0]):

				# Wrap to the beginning
				i0 = 0

	## (raG) = GetGradientVector(self)
	# Get the gradient of error with respect to all learnable network parameters as a vector.
	#
	# * raG - returns error gradient

	def GetGradientVector(self):

		# Start with an empty array
		raG = numpy.array([])

		# For each layer...
		for iLayer in range(self.iLayers):

			# Concatenate the flattened weight gradients
			raG = numpy.concatenate((raG, self.oaStates[iLayer]._raaWg.asarray().ravel()))

		# For each layer...
		for iLayer in range(self.iLayers):

			# Concatenate the flattened bias gradients
			raG = numpy.concatenate((raG, self.oaStates[iLayer]._raBg.asarray().ravel()))

		# Return the error gradient vector
		return(raG)

	## (raW) = GetGradientVector
	# Get all learnable network parameters as a vector.
	#
	# * raW - returns parameter vector

	def GetWeightVector(self):

		# #return(self.raP)
		return(self._raP.asarray().squeeze())

	## SetWeightVector(self, raW)
	# Set all learnable network parameters from a parameter vector.
	#		return(self._raP.asarray().squeeze().copy())

	# * raW - specifies the parameter vector

	def SetWeightVector(self, raW):

		#  self.raP[:] = raW
		self._raP.assign(cudamat.CUDAMatrix(numpy.atleast_2d(raW)))


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
	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9,4)*rScale, numpy.random.randn(4)*rScale)

	# Create layer 1 with random weights and biases
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8,2)*rScale, numpy.random.randn(2)*rScale)

	# Create object
	o = SequenceDecimatingNetwork([oL0,oL1])

	# Compute gradient using backpropagation
	(raG, rError, rRmse) = o.ComputeGradient(raaaX, raaaT)

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
	rScale = 0.001

	# Create layer 0 with random weights and biases
	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9*iMagnify,4*iMagnify)*rScale, numpy.random.randn(4*iMagnify)*rScale)

	# Create layer 1 with random weights and biases
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8*iMagnify,2*iMagnify)*rScale, numpy.random.randn(2*iMagnify)*rScale)

	# Create object
	o0 = SequenceDecimatingNetwork([oL0, oL1])

	# Create layer 0 with random weights and biases
	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9*iMagnify,4*iMagnify)*rScale, numpy.random.randn(4*iMagnify)*rScale)

	# Create layer 1 with random weights and biases
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8*iMagnify,2*iMagnify)*rScale, numpy.random.randn(2*iMagnify)*rScale)

	# Create object
	o1 = SequenceDecimatingNetwork([oL0, oL1])

	# Specify samples in input pattern
	iPatternSamples = 6;

	# Create random input vectors
	raaaX = numpy.random.rand(iPatterns,iPatternSamples,3*iMagnify)

	# Create training vectors from network zero
	raaaT = o0.ComputeOutputs(raaaX)

	# Train network 1 to model network zero
	o1.Train(raaaX, raaaT,  1000, 0.01, 0.0, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))
	o1.Train(raaaX, raaaT, 10000, 0.01, 0.9, lambda iPattern, rError, rRmse: print("iPattern={:6d}, rError={:8.4f}, rRmse={:.6f}".format(iPattern,rError,rRmse)))

cudamat.init()
cudamat.CUDAMatrix.init_random(seed = 42)
TestTrain()