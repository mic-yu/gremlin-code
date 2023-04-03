from netCDF4 import Dataset
import numpy as np

#gpu select
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])

gremlin_path = '../gremlin/gremlin_conus2_dataset.nc'
ds = Dataset(gremlin_path)

nsamples, ny, nx = ds.variables['latitude'].shape
nTrain = 1798
nTest = nsamples - nTrain
nchans = 4
print(nTest)
print(nTrain + nTest)

#training data
#initialize
train = {}
train['Xdata'] = np.zeros((nTrain, ny, nx, nchans))
train['Ydata'] = np.zeros((nTrain, ny, nx))
train['Lat'] = np.zeros((nTrain, ny, nx))
train['Lon'] = np.zeros((nTrain, ny, nx))

#yoink the data
train['Xdata'][:,:,:,0] = ds.variables['GOES_ABI_C07'][0:nTrain]
train['Xdata'][:,:,:,1] = ds.variables['GOES_ABI_C09'][0:nTrain]
train['Xdata'][:,:,:,2] = ds.variables['GOES_ABI_C13'][0:nTrain]
train['Xdata'][:,:,:,3] = ds.variables['GOES_GLM_GROUP'][0:nTrain]
train['Ydata'] = ds.variables['MRMS_REFC'][0:nTrain]
train['Lat'] = ds.variables['latitude'][0:nTrain]
train['Lon'] = ds.variables['longitude'][0:nTrain]


#testing data
#initialize
test = {}
test['Xdata'] = np.zeros((nTest, ny, nx, nchans))
test['Ydata'] = np.zeros((nTest, ny, nx))
test['Lat'] = np.zeros((nTest, ny, nx))
test['Lon'] = np.zeros((nTest, ny, nx))

#yoink the data
test['Xdata'][:,:,:,0] = ds.variables['GOES_ABI_C07'][nTrain:]
test['Xdata'][:,:,:,1] = ds.variables['GOES_ABI_C09'][nTrain:]
test['Xdata'][:,:,:,2] = ds.variables['GOES_ABI_C13'][nTrain:]
test['Xdata'][:,:,:,3] = ds.variables['GOES_GLM_GROUP'][nTrain:]
test['Ydata'] = ds.variables['MRMS_REFC'][nTrain:]
test['Lat'] = ds.variables['latitude'][nTrain:]
test['Lon'] = ds.variables['longitude'][nTrain:]

#saving
data_file = 'gremlin.npz'
np.savez(data_file, Xdata_train=train['Xdata'], Ydata_train=train['Ydata'],
       Xdata_test=test['Xdata'], Ydata_test=test['Ydata'],
       Lat_train=train['Lat'], Lon_train=train['Lon'],
       Lat_test=test['Lat'], Lon_test=test['Lon'])