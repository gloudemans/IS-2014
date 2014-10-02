"""
Provide a collection of methods for accessing the seizure prediction
data.
"""

import numpy
import scipy.io 
import scipy.signal
import os
import sys
import random
import math
import pickle
import pylab

class SeizurePredictionData:
    
	@classmethod
	def LoadMat(cls, sFile):

		# Load one file
		oMat = scipy.io.loadmat(sFile)

		# Find the variable name
		lKeys = [s for s in oMat.keys() if "segment" in s]

		raaData 	= oMat[lKeys[0]]['data'][0,0]
		iElectrodes = oMat[lKeys[0]]['data'][0,0].shape[0]
		iSamples    = oMat[lKeys[0]]['data'][0,0].shape[1]
		rLength     = oMat[lKeys[0]]['data_length_sec'][0,0][0,0]	
		rFrequency  = oMat[lKeys[0]]['sampling_frequency'][0,0][0,0]

		return((raaData, iElectrodes, iSamples, rLength, rFrequency))

	@classmethod
	def ClassOfMat(cls, sFile):

		# Strip any leading path and force to lowercase
		sName = os.path.basename(sFile).lower()

		# If the name contains the string test...
		if sName.count('test'):
			sClass = 'Test'

		elif sName.count('preictal'):
			sClass =  'Preictal'

		else:
			sClass =  'Interictal'

		return(sClass)

	@classmethod
	def PreprocessMatFiles(cls, sSrc, sDst, bDetrend=True, bNormalize=True, iDecimationRatio=1):

		# Set output peak to average signal ratio
		rPeakAverageRatio = 12.

		# If the dource directory doesn't exist...
		if not os.path.exists(sDst):

			# Make it
		    os.makedirs(sDst)

		# If filtering... 
		if iDecimationRatio > 1:

			# Set sinc bandwidth slightly below Nyquist
			rFilterCutoff =  1/(iDecimationRatio*1.3)

			# Filter width in decimated samples
			rWidth = 16

			# Filter width prior to decimation
			iFilterLength = math.ceil(rWidth/rFilterCutoff)

			# Kaiser window beta parameter
			rBeta = 5

			# Construct the windowed sinc filter
			raFilter = numpy.sinc(numpy.linspace(-rWidth/2,rWidth/2,iFilterLength))*numpy.kaiser(iFilterLength, rBeta)

			# pylab.plot(raFilter)
			# pylab.show()
			# return

		# Get all mat filenames
		lFiles = [f for f in os.listdir(sSrc) if(f.endswith('.mat'))]

		# For every matfile...
		for iFile in range(len(lFiles)):

			# Get file name
			f = lFiles[iFile]

			# Load the matfile
			(raaData, iElectrodes, iSamples, rLength, rFrequency) = cls.LoadMat(sSrc + '\\' + f)

			# If detrending...
			if bDetrend:

				# Detrend along time axis
				raaData = scipy.signal.detrend(raaData, axis=1)

			# If decimating...
			if iDecimationRatio > 1:

				# Apply the filter
				raaData = scipy.signal.lfilter(raFilter, 1, raaData, axis=1)

				# Decimate
				raaData = raaData[:,::iDecimationRatio]

			# If normalizing...
			if bNormalize:

				# For each electrode...
				for iElectrode in range(iElectrodes):

					# Remove the column average from this row
					raaData[iElectrode,:] /= raaData[iElectrode,:].std()

				# Scale to specified peak to average ratio
				raaData = numpy.maximum(numpy.minimum(1,raaData/(rPeakAverageRatio)),-1)

				# Transform between zero and one
				raaData = (raaData+1)/2

			# Determine the sample class
			sClass = cls.ClassOfMat(f)

			# Construct the pickle filename
			sDstFile = sDst + '\\' + f[:-3] + 'pkl'

			# Pickle a tuple with fields we want
			oDstFile = open(sDstFile,'wb')
			pickle.dump((raaData.T, rFrequency, sClass), oDstFile)
			oDstFile.close()

			print('{:4d} of {:4d} {}'.format(iFile,len(lFiles),f))

	## SetIndividual
	# Select a particular individual from among the seven possible 
	# individuals: Dog_1, Dog_2, Dog_3, Dog_4, Dog_5, Patient_1, Patient_2
	#
	# * sIndividual - specifies the individual

	def SetIndividual(self, sIndividual):

		# Set the individual
		self.sIndividual = sIndividual

		# Set full path to individual directory
		self.sPath = self.sDirectory + '\\' + sIndividual

		# List of data files
		self.lData = [f for f in os.listdir(self.sPath) if(f.endswith('.mat'))]

		# List of test data files
		self.lTestData = [self.sPath + '\\' + f for f in self.lData if self.ClassOf(f)==-1]

		# List of interictal data files
		self.lInterictalData = [self.sPath + '\\' + f for f in self.lData if self.ClassOf(f)==0]

		# List of preictal data files
		self.lPreictalData = [self.sPath + '\\' + f for f in self.lData if self.ClassOf(f)==1]

		# Combined list of training files
		self.lTrainData = self.lInterictalData + self.lPreictalData

		# Extract		
		(self.raaData, self.iElectrodes, self.iSamples, self.rLength, self.rFrequency) = self.LoadMat(self.lTestData[0])

	def SetPatternSamples(self, iPatternSamples):

		# Set pattern samples
		self.iPatternSamples = iPatternSamples

	def SetValidationSet(self, rValidationFraction):

		self.lRandomInterictals = list(self.lInterictalData)
		self.lRandomPreictals   = list(self.lPreictalData)

		random.shuffle(self.lRandomInterictals)
		random.shuffle(self.lRandomPreictals)

		iValidationInterictals = math.floor(rValidationFraction*len(self.lRandomInterictals))
		iValidationPreictals   = math.floor(rValidationFraction*len(self.lRandomPreictals))

		self.lValidationInterictals = self.lRandomInterictals[:iValidationInterictals]
		self.lValidationPreictals   = self.lRandomPreictals[:iValidationPreictals]
		self.lTrainInterictals      = self.lRandomInterictals[iValidationInterictals:]
		self.lTrainPreictals        = self.lRandomPreictals[iValidationPreictals:]

	def LoadCache(self):

		# Total number of training segments
		iSegments = len(self.lTrainData)
		iSegments = 50

		# Create an array to hold the training 
		self.raaaTrain = numpy.empty((iSegments, self.iElectrodes, self.iSamples), dtype=numpy.float32)
		self.iaTrain   = numpy.empty((iSegments), dtype=numpy.int)

		for iSegment in range(iSegments):

			sFile = self.lRandomPreictals[iSegment];
			self.raaaTrain[iSegment] = self.LoadMat(sFile)[0]
			self.iaTrain[iSegment] = self.ClassOf(sFile)
			print(self.iaTrain[iSegment])

	def GetTestSamples(self, bRestart=False):

		if(bRestart):

			self.iTestSample = 0

		raaData = None
		sFile = None

		if(self.iTestSample<len(self.lTestData)):

			sFile = self.lTestData[self.iTestSample]
			raaData = self.LoadMat(sFile)[0]
			sName = os.path.basename(sFile)
			self.iTestSample+=1

		return(raaData, sName)

		# iFiles = len(self.lTrainData);
		# raaaData = numpy.zeros((iFiles,self.iElectrodes,self.iSamples));
		# for iFile in range(iFiles):
		
		# 	sFile = self.lTrainData[iFile]

		# 	a = self.LoadMat(sFile)[0]

		# 	#print(raaaData[iFile].shape)

		# 	#print(type(a))
		# 	#xx
		# 	raaaData[iFile] = self.LoadMat(sFile)[0]

		# 	print(raaaData.shape)

    ## PreprocessSample
    # Read the specified matfile, process according to the specified options, 
    # and pickle the processed result to the specified file. Processing options 
    # include bias removal, lowpass filtering, subsampling, and level 
    # normalization.

	def PreprocessSample(self, sIn, sOut, bRemoveBias=True, iSubsampleRatio=1, bAntiAlias=False, bNormalize=True):

		(raaData, iElectrodes, iSamples, rLength, rFrequency) = self.LoadMat(sIn)
		print(raaData.shape)

		# # Load one file
		# oMat = scipy.io.loadmat(sFile)

		# # Find the variable name
		# lKeys = [s for s in oMat.keys() if "segment" in s]

		# raaData 	= oMat[lKeys[0]]['data'][0,0]
		# iElectrodes = oMat[lKeys[0]]['data'][0,0].shape[0]
		# iSamples    = oMat[lKeys[0]]['data'][0,0].shape[1]
		# rLength     = oMat[lKeys[0]]['data_length_sec'][0,0][0,0]	
		# rFrequency  = oMat[lKeys[0]]['sampling_frequency'][0,0][0,0]

		# return((raaData, iElectrodes, iSamples, rLength, rFrequency))

		# return((raaData, iElectrodes, iSamples, rLength, rFrequency))


sSrc = 'C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\Dog_1'
sDst = 'C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\400Hz\\Dog_1'
SeizurePredictionData.PreprocessMatFiles(sSrc,sDst,iDecimationRatio=1)
#o.SetIndividual('Dog_1')

#o.PreprocessSample()
# a,b = o.GetTestSample(True)
# print(a.shape)
# print(b)
# a,b = o.GetTestSample()
# print(a[:3,:3])
# print(b)

#o.LoadCache()
#print(o.iElectrodes)

