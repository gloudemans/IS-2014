import urllib.request
import gzip
import struct
import numpy
import pandas

def ReadImages(sFile):

	iJunkBytes = 16
	iImageSize = 28*28 
	oFile = gzip.open(sFile, 'rb')
	oFile.read(iJunkBytes);
	bsImage = oFile.read();
	oFile.close()
	raImage = numpy.fromstring(bsImage, dtype=numpy.uint8)
	raaImage = numpy.reshape(raImage, (raImage.size/iImageSize, iImageSize) )
	return(raaImage)

def ReadLabels(sFile):

	iJunkBytes = 8
	oFile = gzip.open(sFile, 'rb')
	oFile.read(iJunkBytes);
	bsImage = oFile.read();
	oFile.close()
	iaLabels = numpy.fromstring(bsImage, dtype=numpy.uint8)
	return(iaLabels)

# Path to data files on Yann Lecun's web site
sSite = "http://yann.lecun.com/exdb/mnist/";

# Name the required files
sTrainImages = "train-images-idx3-ubyte.gz"
sTrainLabels = "train-labels-idx1-ubyte.gz"
sTestImages  = "t10k-images-idx3-ubyte.gz"
sTestLabels  = "t10k-labels-idx1-ubyte.gz"

# Copy the files locally
urllib.request.urlretrieve (sSite+sTrainImages, sTrainImages)
urllib.request.urlretrieve (sSite+sTrainLabels, sTrainLabels)
urllib.request.urlretrieve (sSite+sTestImages,  sTestImages)
urllib.request.urlretrieve (sSite+sTestLabels,  sTestLabels)

raaTrain = ReadImages(sTrainImages)
raaTest  = ReadImages(sTestImages)
iaTrain = ReadLabels(sTrainLabels)
iaTest  = ReadLabels(sTestLabels)

raa = numpy.concatenate((raaTrain, raaTest))
ia  = numpy.concatenate((iaTrain, iaTest))

df1 = pandas.DataFrame(raa)
df2 = pandas.DataFrame(ia,columns=["labels"])
df3 = pandas.DataFrame(numpy.concatenate((numpy.zeros(iaTrain.size,dtype=numpy.uint8),numpy.ones(iaTest.size,dtype=numpy.uint8))),columns=["subset"])

df  = pandas.concat([df1,df2,df3], axis=1)
df.to_pickle("MNIST.pkl")
print(df["labels"])
