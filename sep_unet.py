from keras.layers import Dense, Activation, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, Flatten, BatchNormalization, UpSampling1D, LeakyReLU, PReLU, Dropout, AveragePooling1D, Reshape, Permute, Add, ELU
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.models import Model, load_model
from keras.constraints import nonneg
from keras import optimizers, losses
from keras import backend as K
from keras.utils import multi_gpu_model

from SelectiveDropout import SelectiveDropout
import sys, getopt
import tensorflow as tf
import h5py
import numpy as np
import time

smooth = 1
kernelSz = 3
batchSz = 128
batchSz = 4096
noGPUs = 4

def normalizeData(smpls):
   row_sums = smpls.mean(axis=1)
   smpls = smpls - row_sums[:, np.newaxis]

   return smpls

def get_msd(noTimepoints=1024,depth=5,features=64,activation_function=PReLU(),lr=1e-4,noGPUs=4,decayrate=0,pDropout=0.25,subsampleData=False):
    layersEncoding = []
    inputs = Input((noTimepoints, 1))
    layers = [inputs]

    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv1D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i)(inputs))
        layers.append(BatchNormalization()(layers[-1]))
        layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv1D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i)(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])

    layers.append(concatenate(layersEncoding))

    # ENCODING LAYER
    layers.append(Conv1D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv1D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))

    layers.append(Conv1D(1,1,activation='linear', padding='same')(layers[-1]))
    o1 = layers[-1]
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(1,activation='sigmoid')(layers[-1]))
    o2 = layers[-1]
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net = Model(layers[0], outputs=[o1,o2])
    u_net.compile(loss=[losses.mean_absolute_error,losses.binary_crossentropy],metrics={'dense_1':'accuracy'}, optimizer=optimizer)
    return u_net


def get_unet(noTimepoints=1024,depth=5,features=64,activation_function=PReLU(),lr=1e-4,noGPUs=4,decayrate=0,pDropout=0.25,subsampleData=False):
    layersEncoding = []
    inputs = Input((noTimepoints, 1))
    layers = [inputs]
    if(subsampleData):
        layers.append(AveragePooling1D(pool_size=8)(layers[-1]))

    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv1D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv1D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])
        layers.append(MaxPooling1D(pool_size=(2))(layers[-1]))

    # ENCODING LAYER
    layers.append(Conv1D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv1D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))

    # UPSAMPLING STREAM
    for i in range(1,depth+1):
        j = depth+1 - i
        layers.append(concatenate([UpSampling1D()(layers[-1]), layersEncoding[-i] ]))
        layers.append(Conv1D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv1D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))

    if(subsampleData):
        layers.append(UpSampling1D(size=8)(layers[-1]))

    layers.append(Conv1D(1,1,activation='linear', padding='same')(layers[-1]))
    o1 = layers[-1]
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(1,activation='sigmoid')(layers[-1]))
    o2 = layers[-1]
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net = Model(layers[0], outputs=[o1,o2])
    u_net.compile(loss=[losses.mean_absolute_error,losses.binary_crossentropy],metrics={'dense_1':'accuracy'}, optimizer=optimizer)
    return u_net

def activateAllDropoutLayers(m):
    ll = [item for item in m.layers if type(item) is SelectiveDropout]
    for ditLayer in ll:
        ditLayer.setDropoutEnabled(true)


def predict_with_uncertainty(model, x, n_iter=10):
    activateAllDropoutLayers(model)
    result = np.zeros((n_iter,) + x.shape)

    for iter in range(n_iter):
        result[iter] = model.predict(x)

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)
    return prediction, uncertainty, result


def applyModel(pDataset, pModel):
  pResult = "res_" + pDataset + "_filteredBy_" + pModel.replace("/","_").replace(".h5","").replace("\\","_") + ".h5"
  print("Loading model: " + pModel)
  with tf.device('/cpu:0'):
    mod = load_model(pModel, custom_objects={"DropoutInTesting": DropoutInTesting})
    mod.summary()
  f = h5py.File(pDataset, "r")
  samples = normalizeData(np.array(f["Bsamples"].value).transpose())
  samples = samples[..., np.newaxis]
  f.close()
  start_time = time.time()
  with tf.device('/cpu:0'):
    samples_pred = mod.predict(samples)
  elapsed_time = time.time() - start_time
  print('elapsed: %.3f s', elapsed_time)
  with h5py.File(pResult,"w") as f:
    d1 = f.create_dataset('samples',data=samples)
    d2 = f.create_dataset('samples_pred',data=samples_pred)


def fit(pData,depth=1,epochs=100,lr=1e-4,model=None,noFeatures=32,activation="sigmoid",batch_size=2**12):
  pModel = "sep_unet_doInTraining_"+str(activation)+"_avPool2_d"+str(depth)+"_f"+str(noFeatures)+"_lr"+str(lr)+"_{epoch:02d}-{val_loss:.6f}.h5"

  print('*** PROCESSING ' + pData + ' with new U-Net compression architecture')
  print('*** U-Net Depth: ' + str(depth) + ' Features ' + str(noFeatures))
  print('*** Checkpointing model to ' + pModel)

  f = h5py.File(pData, "r")
  samples_train = normalizeData(np.array(f["Bsamples"].value))
  noSamples,lengthDatapoint = samples_train.shape
  tau_train = np.array(f["Btau"].value)
  samples_test = normalizeData(np.array(f["Bsamples_test"].value))
  tau_test = np.array(f["Btau_test"].value)
  samples_train = samples_train[..., np.newaxis]
  tau_train = tau_train[..., np.newaxis]
  labels_train = (np.max(tau_train, axis=1) > 0).astype(int)
  labels_test = (np.max(tau_test, axis=1) > 0).astype(int)
  samples_test = samples_test[..., np.newaxis]
  tau_test = tau_test[..., np.newaxis]
  decay_rate = lr / epochs

  if(model is not None):
    unet = load_model(model, custom_objects={"DropoutInTesting": DropoutInTesting})
  else:
    if activation == 'leakyReLU':
        act = LeakyReLU(alpha=0.3)
    elif activation == 'ELU':
        act = ELU(alpha=1.0)
    elif activation == 'sigmoid':
        act = Activation('sigmoid')
               
    unet = get_unet(noTimepoints=lengthDatapoint,depth=depth,features=noFeatures,noGPUs=1,decayrate=decay_rate,lr=lr,activation_function=act,subsampleData=True)
    
  #unet.summary()
  checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list=[]
  unet.fit(samples_train, [tau_train,labels_train], batch_size=batch_size, epochs=epochs, validation_data=(samples_test, [tau_test, labels_test]),
            verbose=2, callbacks=callbacks_list)
  
  return unet


def main(argv):
   pData = 'snklb_Raw_muSigmaVariation.mat'
   pModel = None
   doLearning = False
   depth = 2
   noEpochs = 1000
   lr = 1e-3
   noFeatures = 32
   try:
      opts, args = getopt.getopt(argv,"hld:i:m:e:f:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('sep_unet.py -l -d depth -f noFeatures -i inputDataset -m modelToApply')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('sep_unet.py -l -d depth -i inputDataset -m model to apply')
         sys.exit()
      elif opt in ("-l", "--doLearning"):
         doLearning = True
      elif opt in ("-d", "--depth"):
         depth = int(arg)
      elif opt in ("-i", "--inputDataset"):
         pData = arg
      elif opt in ("-m", "--modelPath"):
         pModel = arg
      elif opt in ("-e", "--epochs"):
         noEpochs = arg
      elif opt in ("-f", "--features"):
         noFeatures = int(arg)
   if(doLearning == False):
     applyModel(pData, pModel)
   else:
     fit(pData=pData, depth=depth,epochs=noEpochs,model=pModel,lr=lr,noFeatures=noFeatures)

if __name__ == '__main__':
  main(sys.argv[1:])
