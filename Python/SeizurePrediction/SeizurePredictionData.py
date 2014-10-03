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
#import pylab

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
	def ClassFromName(cls, sFile):

		# Strip any leading path and force to lowercase
		sName = os.path.basename(sFile).lower()

		# If the name contains the string test...
		if sName.count('test'):
			sClass = 'Test'

		elif sName.count('preictal'):
			sClass = 'Preictal'

		else:
			sClass = 'Interictal'

		return(sClass)

	@classmethod
	def PreprocessMatFiles(cls, sSrc, sDst, bDetrend=True,  rSampleFrequency=400):

		# Always normalize the data
		bNormalize = True

		# Always use the same peak to average signal ratio
		rPeakAverageRatio = 12

		# If the dource directory doesn't exist...
		if not os.path.exists(sDst):

			# Make it
		    os.makedirs(sDst)

		# Get all mat filenames
		lFiles = [f for f in os.listdir(sSrc) if(f.endswith('.mat'))]

		# For every matfile...
		for iFile in range(len(lFiles)):

			# Get file name
			f = lFiles[iFile]

			# Load the matfile
			(raaData, iElectrodes, iSamples, rLength, rFrequency) = cls.LoadMat(sSrc + '\\' + f)

			# Compute the nearest integer decimation ratio
			iDecimationRatio = int(round(rFrequency/rSampleFrequency))

			# If detrending...
			if bDetrend:

				# Detrend along time axis
				raaData = scipy.signal.detrend(raaData, axis=1).astype(numpy.float32)

			# If decimating...
			if iDecimationRatio > 1:

				# Decimate using 8th order chebyshev IIR
				raaData = scipy.signal.decimate(raaData, iDecimationRatio, axis=1).astype(numpy.float32)

			# If normalizing...
			if bNormalize:

				# For each electrode...
				for iElectrode in range(iElectrodes):

					# Remove the column average from this row
					raaData[iElectrode,:] /= raaData[iElectrode,:].std()

				# Scale to specified peak to average ratio
				raaData = numpy.maximum(numpy.minimum(1,raaData/(rPeakAverageRatio)),-1).astype(numpy.float32)

				# Transform between zero and one
				raaData = (raaData+1)/2

			# Determine the sample class
			sClass = cls.ClassFromName(f)

			# Construct the pickle filename
			sDstFile = sDst + '\\' + f[:-3] + 'pkl'

			# Pickle a tuple with fields we want
			pickle.dump((raaData.T, rFrequency, sClass), open(sDstFile,'wb'))

			print('{:4d} of {:4d} {}'.format(iFile,len(lFiles),f))

	@classmethod
	def PreprocessDataset(cls, sSrc, sDst, bDetrend=True, rSampleFrequency=400):

		# Get subdirectories of the specified source directory
		lSubs = [f for f in os.listdir(sSrc) if os.path.isdir(os.path.join(sSrc, f))]

		# For each subdirectory...
		for s in lSubs:

			# Preprocess the matfiles
			cls.PreprocessMatFiles(os.path.join(sSrc, s), os.path.join(sDst, s), bDetrend, rSampleFrequency)

	@classmethod
	def MakeBatches(cls, sSrc, sDst, iBatches, iBatchSamples, iSubsamples):

		# If the dource directory doesn't exist...
		if not os.path.exists(sDst):

			# Make it
		    os.makedirs(sDst)

		# Get all training pkl filenames
		lFiles = [f for f in os.listdir(sSrc) if(f.endswith('.pkl') and (not cls.ClassFromName(f)=='Test'))]

		# Report status
		print('Loading {} ...'.format(sSrc))

		# For every training file...
		for iFile in range(len(lFiles)):

			# Get file name
			f = lFiles[iFile]

			# Get fields
			(raaData, rFrequency, sClass) = pickle.load(open(os.path.join(sSrc, f),'rb'))

			# If this is the first file...
			if not iFile:

				# Create arrays
				iRows = raaData.shape[0]
				iCols = raaData.shape[1]
				raaaData = numpy.empty((len(lFiles), iRows, iCols))
				iaClass  = numpy.empty((len(lFiles)))

			# Store the data
			raaaData[iFile,:,:] = raaData

			# Store class flag
			iaClass[iFile] = sClass =='Preictal'

		print('Loading Complete...')

		for iBatch in range(iBatches):

			raaSample = numpy.empty((iBatchSamples, iSubsamples*iCols), dtype=numpy.float32)
			iaSample = numpy.empty((iBatchSamples), dtype = numpy.int)

			sBatch =  os.path.join(sDst, 'Batch_{:04d}.pkl'.format(iBatch))

			for iBatchSample in range(iBatchSamples):

				iFile   = random.randrange(len(lFiles))
				iOffset = random.randrange(iRows-iSubsamples)

				raaSample[iBatchSample,:] = raaaData[iFile,iOffset:iOffset+iSubsamples,:].flatten()
				iaSample[iBatchSample] = iaClass[iFile]

				if(not (iBatchSample%10000) ):
					print('sFile={:s}, iBatchSample={:6d}'.format(os.path.basename(sBatch), iBatchSample))

			# Pickle the batch
			print(iaSample[:20])
			pickle.dump((raaSample, iaSample), open(sBatch,'wb'))


	@classmethod
	def MakeAllBatches(cls, sSrc, sDst, iBatches, iBatchSamples, iSamples):

		# Get subdirectories of the specified source directory
		lSubs = [f for f in os.listdir(sSrc) if os.path.isdir(os.path.join(sSrc, f))]

		# For each subdirectory...
		for s in lSubs:

			# Preprocess the matfiles
			cls.MakeBatches(os.path.join(sSrc, s), os.path.join(sSrc, s, sDst), iBatches, iBatchSamples, iSamples)

	@classmethod
	def Bulk(cls):

		bDetrend = True
		sDataset = 'C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\'

		cls.PreprocessDataset(os.path.join(sDataset,'Raw'), os.path.join(sDataset, '20Hz'), True,  20.0)
		cls.PreprocessDataset(os.path.join(sDataset,'Raw'), os.path.join(sDataset,'400Hz'), True, 400.0)

	@classmethod
	def Bulk2(cls):

		iBatches = 100
		iSamples = 16;
		iBatchSamples = 100000
		sDataset = 'C:\\Users\\Mark\\Documents\\GitHub\\IS-2014\\Datasets\\Kaggle Seizure Prediction Challenge\\20Hz'

		cls.MakeAllBatches(sDataset, 'Layer_0', iBatches, iBatchSamples, iSamples)


SeizurePredictionData.Bulk()