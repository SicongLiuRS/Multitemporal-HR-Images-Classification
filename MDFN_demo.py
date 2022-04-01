from HyperFunctions import*
import time
import scipy.io as sio
from keras.utils import np_utils
from model import MDFNmodel
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

# 模型参数
dataID = 1
wsize =  8
randtime = 10
nb_epoch = 20
batch_size = 128
time_step = 4
s1s2 = 1
model_name = 'MDFN_'
OASpatial = np.zeros((5+3  ,randtime+1))
#********************************************************************
for r in range(0,randtime):

    X, X_train, X_test, XP, XP_train, XP_test, Y, Y_train, Y_test, train_indexes =\
        MultispectralSamples(dataID=dataID, timestep= time_step, w=wsize, israndom=False, s1s2=s1s2, random=r)

    # sio.savemat('./GF_Data/train_indexes_'+ str(r+1) +'.mat', {'train_indexes': train_indexes})

    img_rows, img_cols = wsize, wsize
    nb_features = X.shape[-1]

    Y = Y - 1
    Y_train = Y_train - 1
    Y_test = Y_test - 1
    nb_classes = Y_train.max()+1

    XP = np.asarray(XP, dtype=np.float32)
    XP_train = np.asarray(XP_train, dtype=np.float32)
    XP_test  = np.asarray(XP_test, dtype=np.float32)

    XP_T       = np.reshape(XP, [XP.shape[0], img_rows, img_cols, nb_features, time_step])
    XP_train_T = np.reshape(XP_train, [XP_train.shape[0], img_rows, img_cols, nb_features, time_step])
    XP_test_T  = np.reshape(XP_test,  [XP_test.shape[0],  img_rows, img_cols, nb_features, time_step])

    XP_S       = np.zeros([XP.shape[0], img_rows, img_cols, time_step, nb_features])
    XP_train_S = np.zeros([XP_train.shape[0], img_rows, img_cols, time_step, nb_features])
    XP_test_S  = np.zeros([XP_test.shape[0], img_rows, img_cols, time_step, nb_features])

    for i in range(0, nb_features):
        XP_S[:, :, :, 0, i] = XP[:,:,:,i]
        XP_S[:, :, :, 1, i] = XP[:, :, :, i + 4]
        XP_S[:, :, :, 2, i] = XP[:, :, :, i + 8]
        # XP_S[:, :, :, 3, i] = XP[:, :, :, i + 12]

        XP_train_S[:, :, :, 0, i] = XP_train[:,:,:,i]
        XP_train_S[:, :, :, 1, i] = XP_train[:, :, :, i + 4]
        XP_train_S[:, :, :, 2, i] = XP_train[:, :, :, i + 8]
        # XP_train_S[:, :, :, 3, i] = XP_train[:, :, :, i + 12]

        XP_test_S[:, :, :, 0, i]  = XP_test[:,:,:,i]
        XP_test_S[:, :, :, 1, i]  = XP_test[:, :, :, i + 4]
        XP_test_S[:, :, :, 2, i]  = XP_test[:, :, :, i + 8]
        # XP_test_S[:, :, :, 3, i] = XP_test[:, :, :, i + 12]
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)
#*************************SSFN*******************************************
    #模型训练
    start = time.time()
    model = MDFNmodel.MDFN(nb_classes, nb_features, img_rows, img_cols, time_step)
    histloss = model.fit([X_train, XP_train_T, XP_train_S], [y_train], nb_epoch=nb_epoch, batch_size=batch_size,
                         verbose=1, shuffle=True)
    losses = histloss.history
    #模型预测
    PredictLabel = model.predict([X_test, XP_test_T, XP_test_S], verbose=1).argmax(axis=-1)
    end = time.time()
    Runtime = end - start

    # 精度评价
    PredictLabel = np.asarray(PredictLabel, dtype=np.int)
    OA,Kappa,ProducerA = CalAccuracy(PredictLabel,Y_test[:,0])
    OASpatial[0:nb_classes,r] = ProducerA
    OASpatial[-3,r] = OA
    OASpatial[-2,r] = Kappa
    OASpatial[-1,r] = Runtime
    # 结果输出
    print('rand',r+1,'test accuracy:', OA*100)
    # 输出全图
    # MapLabel = model.predict([XP_T, XP_S], verbose=1).argmax(axis=-1)
    # MapLabel = model.predict([X], verbose=1).argmax(axis=-1)
    MapLabel = model.predict([X, XP_T, XP_S], verbose=1).argmax(axis=-1)
    MapLabel = np.asarray(MapLabel, dtype=np.int)
    X_result = DrawResult(MapLabel, dataID)
    save_fn = './result/Tree/'+ model_name + str(r+1) +'_classmap.png'
    plt.imsave(save_fn, X_result)
    # 输出mask图
    X_result1 = X_result
    no_lines, no_columns, no_bands = X_result1.shape[0], X_result1.shape[1], X_result1.shape[2]
    X_result1 = np.reshape(X_result1, (no_lines * no_columns, no_bands))
    X_result1[np.where(Y == -1), 0] = 0
    X_result1[np.where(Y == -1), 1] = 0
    X_result1[np.where(Y == -1), 2] = 0
    X_result1 = np.reshape(X_result1, (no_lines, no_columns, no_bands))
    save_fn = './result/Tree/' + model_name + str(r+1) + '_classmap_mask.png'
    plt.imsave(save_fn, X_result1)
#结果另存
OASpatial[:, -1] = np.mean(OASpatial[:, 0:-1], axis=1)
print(OASpatial[:, -1])
save_mat = './result/Tree/'+ model_name +'_OA.mat'
sio.savemat(save_mat, {'CNN_OA': OASpatial})
