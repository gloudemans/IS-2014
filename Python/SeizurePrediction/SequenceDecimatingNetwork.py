import math
import numpy

## SequenceDecimatingNetwork
# This class implements a sequence decimating network.
#
# * SequenceDecimatingNetwork
# * ComputeOutputs
# * ComputeGradient
# * GetWeightVector
# * SetWeightVector
# * Train

class SequenceDecimatingNetwork:

	# Small offset to prevent log underflow when computing 
	rEps = 1e-10

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

	def ComputeOutputs(self, raaaX, bComputeDerivatives=False):

		(iPatterns, iSamples, iFeatures) = raaaX.shape 

		# Save input states
		self.oaStates[0].raaX = numpy.reshape(numpy.copy(raaaX),(iPatterns*iSamples,iFeatures))

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

			# If this derivative is needed for backpropagation
			if(bComputeDerivatives and (iLayer<self.iLayers-1)):

				# Compute logistic'(x) activation derivitive
				self.oaStates[iLayer+1].raaD = (1-self.oaStates[iLayer+1].raaX)*self.oaStates[iLayer+1].raaX

		return(self.oaStates[self.iLayers].raaX)

	def ComputeGradient(self, raaX, raaT):

		raaY = self.ComputeOutputs(raaX, bComputeDerivatives=True)

		raY = raaY.flatten()
		raT = raaT.flatten()
		rError = -numpy.mean(raT*numpy.log(raY+self.rEps) + (1-raT)*numpy.log(1-raY+self.rEps))
		rRmse  = numpy.sqrt(numpy.mean((raT-raY)**2))

		# Compute output layer error
		raaE = raaY - raaT

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

		raG = self.GetGradientVector()

		return((raG, rError, rRmse))

	def ComputeGradientNumerical(self, raaX, raaT, rDelta=1e-6):

		rEps = 1e-10

		raY = self.ComputeOutputs(raaX).flatten()
		raT = raaT.flatten()
		rError = -numpy.sum(raT*numpy.log(raY+rEps) + (1-raT)*numpy.log(1-raY+rEps))

		raW = self.GetWeightVector()
		raG = numpy.zeros(raW.shape)
		raWd = numpy.copy(raW)


		for k in range(len(raW)):
			raWd[k] += rDelta
			self.SetWeightVector(raWd)
			raWd[k] = raW[k]
			raY0 = self.ComputeOutputs(raaX).flatten()
			rError0 = -numpy.sum(raT*numpy.log(raY0+rEps) + (1-raT)*numpy.log(1-raY0+rEps))
			raG[k] = (rError0-rError)/rDelta

		self.SetWeightVector(raW)

		self.ComputeOutputs(raaX, True)

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

	def Train(self, raaaX, raaT, iSamples, rRate, rMomentum):

		iSample = 0
		iBatch  = 100
		i0 = 0
		raW = self.GetWeightVector()

		while(iSample<iSamples):

			i1 = min(i0 + min(iBatch,iSamples-iSample), raaaX.shape[0])
			raaXs = raaaX[i0:i1,:,:]
			raaTs = raaT[i0:i1,:]
			iSample += i1-i0
			i0 = i1;

			(raG, rError, rRmse) = self.ComputeGradient(raaXs, raaTs)

			print("rError={:8.4f}, rRmse={:.6f}".format(rError,rRmse))

			raW -= rRate*raG
			self.SetWeightVector(raW)

			if(i0==raaaX.shape[0]):
				i0 = 0

def TestGradient():

	iPatterns = 5;
	iPatternSamples = 6;


	# Create random input vectors
	raaX = numpy.random.randn(iPatterns,iPatternSamples,3)*.1

	# Create random target vectors
	raaT = numpy.random.randn(iPatterns,2)*.1

	# Create layer 0 with random weights and biases
	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9,4)*.1, numpy.random.randn(4)*.1)

	# Create layer 1 with random weights and biases
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8,2)*.1, numpy.random.randn(2)*.1)

	# Create object
	o = SequenceDecimatingNetwork([oL0,oL1])

	# Compute gradient using backpropagation
	(raG, rError, rRmse) = o.ComputeGradient(raaX, raaT)

	# Compute gradient numerically
	raGn = o.ComputeGradientNumerical(raaX, raaT)

	# Compute error measure
	rError = 2*sum((raG-raGn)**2)/sum(0.5*(raG**2+raGn**2))

	# Report error measure
	print("Gradient Test: rError={:f}".format(rError))

def TestTrain():

	iPatterns = 1000
	rScale = .01

	raaX = numpy.random.randn(iPatterns,6,3)*rScale

	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9,4)*rScale, numpy.random.randn(4)*rScale)
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8,2)*rScale, numpy.random.randn(2)*rScale)

	o0 = SequenceDecimatingNetwork([oL0, oL1])


	oL0 = SequenceDecimatingNetwork.Layer(3, numpy.random.randn(9,4)*rScale, numpy.random.randn(4)*rScale)
	oL1 = SequenceDecimatingNetwork.Layer(2, numpy.random.randn(8,2)*rScale, numpy.random.randn(2)*rScale)

	o1 = SequenceDecimatingNetwork([oL0, oL1])

	raaT = o0.ComputeOutputs(raaX)

	o1.Train(raaX, raaT, 1000, 0.01, 0.5)

TestTrain()