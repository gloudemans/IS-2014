import numpy
import scipy.io 
import os
import sys
import random
import math

class DataProxy:
    
	pass

	## DataProxy
	# Create a new DataProxy using the specified database path and cache size.
	#
	# * sDirectory - specifies the path to the top level seizure prediction 
	#   dataset directory
	# * sIndividual - specifies the individual dataset (Dog_1...5 or Patient_1...2)
	# * rValidationFraction - specifies the fraction of training samples to reserve for
	#   validation
	# * iCacheSegments - specifies the maximum number of segments to retain in memory 

	def __init__(self, sDirectory, sIndividual='Dog_1', rValidationFraction=0.2, iCacheSegments=256):

		# Set database directory
		self.sDirectory = sDirectory

		# Set cache segments
		self.iCacheSegments = iCacheSegments

		# Default state
		self.SetIndividual('Dog_1')

		# Default state
		self.SetValidationSet(rValidationFraction)

	## 
	# Load the .mat file specified by sFile and return the
	# data array, the number of electrodes, the number of samples,
	# the length, and the sampling frequency as a tuple.

	def LoadMat(self, sFile):

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

	## SegmentClass
	# Return class given the specified mat filename.
	# The returned class is an integer 1 for preictal, 0 for interictal and
	# -1 for test.
	#
	# * sFile - specifies the filename

	def ClassOf(self, sFile):

		# Strip any leading path and force to lowercase
		sName = os.path.basename(sFile).lower()

		# If the name contains the string test...
		if sName.count('test'):

			# Return sentinel indicating test
			iClass = -1

		# If the name contains the work preictal...
		elif sName.count('preictal'):

			# Class is 1
			iClass =  1

		else:

			# Class is 0
			iClass =  0

		return(iClass)

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



o = DataProxy('C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge')   

a,b = o.GetTestSample(True)
print(a.shape)
print(b)
a,b = o.GetTestSample()
print(a[:3,:3])
print(b)

#o.LoadCache()
#print(o.iElectrodes)

