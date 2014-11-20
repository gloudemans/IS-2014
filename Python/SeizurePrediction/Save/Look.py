import pickle 
import pylab

sPath = 'C:/Users/Mark/Documents/GitHub/IS-2014/Datasets/Kaggle Seizure Prediction Challenge/400Hz/Dog_1/Layer_0/Batch_0001.pkl'

(raaTrain, iaTrain, raaTest, iaTest) = pickle.load( open( sPath, "rb" ) )

#pylab.plot(raaTrain[:,2::16])
#pylab.show()

print(raaTrain[:].std())

