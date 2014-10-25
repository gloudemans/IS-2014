import math
import numpy

## SequenceDecimatingNetwork
# This class implements a sequence decimating network.
#  It provides methods to
# read and write the network weights as a vector and to compute the gradient
# of error with respect to network weights.
#
# * SequenceDecimatingNetwork
# * ComputeOutputs
# * ComputeGradient
# * GetWeightVector
# * SetWeightVector
# * Train

class SequenceDecimatingNetwork:

	## Layer
	# This contains the parameters that descibe a network layer.

	class Layer:

		def __init__(self, iDecimation, raaW, raB):

			# Number of input samples per output summary			
			self.iDecimation = iDecimation

			# Connection weights from input to output [iInputs, iOutputs] 
			self.raaW = raaW

			# Biases for each output [iOutputs]
			self.raB  = raB

	## State
	# State variables used during output computation and
	# gradient computation for a layer.

	class State:

		def __init__(self):

			# Layer inputs [iSamples, iInputs]
			self.raaX = []

			# Layer derivatives [iSamples, iInputs]
			self.raaD = []

			# Weight gradients [iInputs, iOutputs]
			self.raaWg = []

			# Bias gradients [iOutpus]
			self.raaBg = []		

	def __init__(self, oaLayers):

		# Number of connection layers in the network
		self.iLayers = len(oaLayers)

		# List of connection layers
		self.oaLayers = oaLayers

		# List of states
		self.oaStates = []

		# For each layer...
		for iLayer in range(self.iLayers):

			# Create a state 
			self.oaStates.append( SequenceDecimatingNetwork.State())

		# Need an extra state for the network output
		self.oaStates.append( SequenceDecimatingNetwork.State())

	def ComputeOutputs(self, raaaP, bComputeDerivatives=False):

		#print(self.GetWeightVector())

		# Measure the input array
		(iPatterns, iSamples, iFeatures) = raaaP.shape

		# Reinterpret shape to allow efficient matrix computations
		self.oaStates[0].raaX = numpy.reshape(numpy.copy(raaaP), (iPatterns*iSamples, iFeatures))

		# For each layer...
		for iLayer in range(self.iLayers):

			# Measure layer input
			(iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape

			# Aggregate features from multiple samples
			iFeatures *= self.oaLayers[iLayer].iDecimation

			# Decimate the layer
			self.oaStates[iLayer].raaX = numpy.reshape(self.oaStates[iLayer].raaX, (-1, iFeatures))
	
			# Compute activation function input
			self.oaStates[iLayer+1].raaX = numpy.dot(self.oaStates[iLayer].raaX, self.oaLayers[iLayer].raaW)

			# For each sample...
			for iSample in range(self.oaStates[iLayer+1].raaX.shape[0]):

				# Add bias
				self.oaStates[iLayer+1].raaX[iSample,:] += self.oaLayers[iLayer].raB

			# Compute logistic(x) activation
			self.oaStates[iLayer+1].raaX = 1./(1+numpy.exp(-self.oaStates[iLayer+1].raaX))

			print(numpy.sum(self.oaStates[iLayer+1].raaX))

			# If this derivative is needed for backpropagation
			if(bComputeDerivatives and (iLayer<self.iLayers-1)):

				# Compute logistic'(x) activation derivitive
				self.oaStates[iLayer+1].raaD = (1-self.oaStates[iLayer+1].raaX)*self.oaStates[iLayer+1].raaX

		return(self.oaStates[self.iLayers].raaX)

	def ComputeGradient(self, raaaP, raaT):

		raaY = self.ComputeOutputs(raaaP, bComputeDerivatives=True)

		# Compute output layer error
		raaE = raaY - raaT

		# For each layer...
		for iLayer in range(self.iLayers-1,-1,-1):

			# Measure the layer input
			(iSamples, iFeatures) = self.oaStates[iLayer].raaX.shape

			# Compute the gradient of error with respect to weight
			self.oaStates[iLayer].raaWg = numpy.dot(self.oaStates[iLayer].raaX.T, raaE)/iSamples

			# Compute gradient of error with respect to bias
			self.oaStates[iLayer].raBg = numpy.mean(raaE,0)

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

		raG = self.GetGradientVector()

		return(raG)

	def ComputeGradientNumerical(self, raaaP, raaT, rDelta=1e-6):

		rEps = 1e-10

		raY = self.ComputeOutputs(raaaP).flatten()
		raT = raaT.flatten()
		rError = -numpy.mean(raT*numpy.log(raY+rEps) + (1-raT)*numpy.log(1-raY+rEps))

		raW = self.GetWeightVector()
		raG = numpy.zeros(raW.shape)
		raWd = numpy.copy(raW)


		for k in range(len(raW)):
			raWd[k] += rDelta
			self.SetWeightVector(raWd)
			raWd[k] = raW[k]
			raY0 = self.ComputeOutputs(raaaP).flatten()
			rError0 = -numpy.mean(raT*numpy.log(raY0+rEps) + (1-raT)*numpy.log(1-raY0+rEps))
			raG[k] = (rError0-rError)/rDelta

		self.SetWeightVector(raW)

		self.ComputeOutputs(raaaP, True)

		return(raG)

	def GetGradientVector(self):

		raG = numpy.array([])
		for iLayer in range(self.iLayers):
			raG = numpy.concatenate((raG, self.oaStates[iLayer].raaWg.ravel()))
			raG = numpy.concatenate((raG, self.oaStates[iLayer].raBg.ravel()))
		return(raG)

	def GetWeightVector(self):

		raW = numpy.array([])
		for iLayer in range(self.iLayers):
			raW = numpy.concatenate((raW, self.oaLayers[iLayer].raaW.ravel()))
			raW = numpy.concatenate((raW, self.oaLayers[iLayer].raB.ravel()))
		return(raW)

	def SetWeightVector(self, raW):

		# Keep this or you'll be sorry!
		raW = numpy.copy(raW)

		iBase = 0
		for iLayer in range(self.iLayers):

			self.oaLayers[iLayer].raaW = numpy.reshape(raW[iBase:iBase+self.oaLayers[iLayer].raaW.size],self.oaLayers[iLayer].raaW.shape)
			iBase += self.oaLayers[iLayer].raaW.size

			self.oaLayers[iLayer].raB  = numpy.reshape(raW[iBase:iBase+self.oaLayers[iLayer].raB.size], self.oaLayers[iLayer].raB.shape)
			iBase += self.oaLayers[iLayer].raB.size

def Test():

	# Create network designed to 1000 patterns containing 100 samples with 15 features each.
	# The first layer decimates by 5 resulting in 75 input features and generates 100 output features
	# The second layer decimates by 20 resulting in 2000 input features and 1 output feature

	# raaX = numpy.random.randn(2000,100,15)
	# raaT = numpy.random.randn(2000,2)
	# oL0 = SequenceDecimatingNetwork.Layer(5,  numpy.random.randn(75,100), numpy.random.randn(100))
	# oL1 = SequenceDecimatingNetwork.Layer(20, numpy.random.randn(2000,2), numpy.random.randn(2))

	raaX = numpy.random.randn(5,6,3)*.1
	raaT = numpy.random.randn(5,2)*.1
	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9,4)*.1, numpy.random.randn(4)*.1)
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8,2)*.1, numpy.random.randn(2)*.1)

	o = SequenceDecimatingNetwork([oL0,oL1])
	raG = o.ComputeGradient(raaX, raaT)
	raGn = o.ComputeGradientNumerical(raaX, raaT)
	print(raG[0:4])
	print(raGn[0:4])

	rE = max(numpy.abs(raG-raGn))
#print(rE)
#	print(raW.shape)

Test()