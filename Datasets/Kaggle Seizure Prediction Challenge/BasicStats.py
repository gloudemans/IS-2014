import scipy.io 
import os
import sys

# Get a list of the subdirectories
lSubs = [f for f in os.listdir() if os.path.isdir(f)]

iTotalTest       = 0
iTotalInterictal = 0
iTotalPreictal   = 0

log = open('BasicStats.csv','wt')

print('Dir,Interictal Files,Preictal Files,Test Files,Electrodes,Samples,Length (S),Frequency (Hz),File Bytes(MB),Dir Bytes(GB),Consistent',file=log)

iDataBytes = 0;
for sSub in lSubs:

	lMats = [f for f in os.listdir(sSub) if(f.endswith('.mat'))]

	iTest       = len([mat for mat in lMats if mat.count('test')])
	iInterictal = len([mat for mat in lMats if mat.count('interictal')])
	iPreictal   = len([mat for mat in lMats if mat.count('preictal')])

	# Load the first matfile
	a = scipy.io.loadmat(sSub + '\\' + lMats[0])
	iSize = os.path.getsize(sSub + '\\' + lMats[0])

	# Find the variable name
	key = [s for s in a.keys() if "segment" in s]

	# Extract		
	iElectrodes = a[key[0]]['data'][0,0].shape[0]	
	iSamples    = a[key[0]]['data'][0,0].shape[1]	
	rLength     = a[key[0]]['data_length_sec'][0,0][0,0]	
	rFrequency  = a[key[0]]['sampling_frequency'][0,0][0,0]

	bConsistent =  True;
	iTotalSize = 0;
	for mat in lMats:

		# Load the first matfile
		a = scipy.io.loadmat(sSub + '\\' + mat)

		# Find the variable name
		key = [s for s in a.keys() if "segment" in s]

		bConsistent &= (iElectrodes == a[key[0]]['data'][0,0].shape[0])	
		bConsistent &= (iSamples    == a[key[0]]['data'][0,0].shape[1])	
		bConsistent &= (rLength     == a[key[0]]['data_length_sec'][0,0][0,0])	
		bConsistent &= (rFrequency  == a[key[0]]['sampling_frequency'][0,0][0,0])

		iTotalSize += os.path.getsize(sSub + '\\' + mat)

		print('.',end='')
		sys.stdout.flush()

	print();


	iSize = iTotalSize / len(lMats)


	print('{:10s}, iInterictal={:4d}, iPreictal={:4d}, iTest={:4d}, iElectrodes={:2d}, iSamples={:7d}, rLength={:.0f}, rFrequency={:7.2f}, iBytes={:.3g}, iTotalBytes={:.3g}, bConsistent={}'.format(
		sSub, iInterictal, iPreictal, iTest, iElectrodes, iSamples, rLength, rFrequency, iSize, iTotalSize, bConsistent))
	print('{},{},{},{},{},{},{:.0f},{:.2f},{:.3g},{:.3g},{}'.format(sSub, iInterictal, iPreictal, iTest, iElectrodes, iSamples, rLength, rFrequency, iSize/1e6, iTotalSize/1e9, bConsistent),file=log)

	iTotalTest       += iTest
	iTotalInterictal += iInterictal
	iTotalPreictal   += iPreictal
	iDataBytes       += iTotalSize

print('iTotalInterictal={:4d}, iTotalPreictal={:4d}, iTotalTest={:4d}, iTotal={:5d}, DataBytes={:.3g} (GB)'.format(iTotalInterictal, iTotalPreictal, iTotalTest, iTotalInterictal+iTotalPreictal+iTotalTest, iDataBytes/1e9))
print('TotalInterictal={:4d}, TotalPreictal={:4d}, TotalTest={:4d}, Total={:5d}, DataBytes={:.3g} (GB)'.format(iTotalInterictal, iTotalPreictal, iTotalTest, iTotalInterictal+iTotalPreictal+iTotalTest, iDataBytes/1e9),file=log)

log.close()