from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model


def MDFN(nb_classes, nb_features, img_rows, img_cols,time_step):

    LSTMInput = layers.Input(shape=(time_step, nb_features), name='LSTMInput')
    LSTMSpectral = layers.LSTM(128,  return_sequences=True, activation='relu', name='LSTM1')(LSTMInput)
    LSTMSpectral = layers.LSTM(128, return_sequences=True, activation='relu', name='LSTM2')(LSTMSpectral)
    LSTMDense = layers.LSTM(128, return_sequences=False, activation='relu', name='LSTM3')(LSTMSpectral)

    CNNTInput = layers.Input(shape=[img_rows, img_cols, nb_features, time_step], name='CNNTInput')
    CONV3DT_1 = layers.Conv3D(64, (1, 1, 4), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3T_1')(CNNTInput)
    CONV3DT_2 = layers.Conv3D(64, (3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3T_2')(CNNTInput)
    Fusion3D_1 = layers.Concatenate()([CONV3DT_1, CONV3DT_2])
    Fusion3D_1 = layers.BatchNormalization(axis=-1)(Fusion3D_1)

    CNNSInput = layers.Input(shape=[img_rows, img_cols, time_step, nb_features], name='CNNSInput')
    CONV3DS_1 = layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3S_1')(CNNSInput)
    CONV3DS_2 = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3S_2')(CNNSInput)
    Fusion3D_2 = layers.Concatenate()([CONV3DS_1, CONV3DS_2])
    Fusion3D_2 = layers.BatchNormalization(axis=-1)(Fusion3D_2)

    Fusion3D_1 = layers.Reshape((int(Fusion3D_1.shape[1]), int(Fusion3D_1.shape[2]), int(Fusion3D_1.shape[3] * Fusion3D_1.shape[4])))(Fusion3D_1)
    Fusion3D_2 = layers.Reshape((int(Fusion3D_2.shape[1]), int(Fusion3D_2.shape[2]), int(Fusion3D_2.shape[3] * Fusion3D_2.shape[4])))(Fusion3D_2)

    Fusion2D_1 = layers.Concatenate()([Fusion3D_1, Fusion3D_2])
    Fusion2D_1 = layers.BatchNormalization(axis=-1)(Fusion2D_1)

    CONV2D_1 = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='CONV2D_1')(Fusion2D_1)
    CONV2D_2 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='CONV2D_2')(Fusion2D_1)
    Fusion2D_2 = layers.Concatenate()([CONV2D_1, CONV2D_2])
    Fusion2D_2 = layers.BatchNormalization(axis=-1)(Fusion2D_2)
    CNNDense = layers.Flatten(name='FLATTEN')(Fusion2D_2)

    Fusion = layers.Concatenate()([LSTMDense, CNNDense])
    Fusion = layers.BatchNormalization(axis=-1)(Fusion)
    FusionDENSE = layers.Dense(128, activation='relu', name='FusionDENSE')(Fusion)

    FusionSOFTMAX = layers.Dense(nb_classes, activation='softmax', name='FusionSOFTMAX')(FusionDENSE)

    model = Model(input=[LSTMInput, CNNTInput, CNNSInput], output=[FusionSOFTMAX])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 网络可视化
    # plot_model(model, to_file='CACNN.png')
    model.summary()
    return model


def LSTM(nb_classes, nb_features, time_step):

    LSTMInput = layers.Input(shape=(time_step, nb_features), name='LSTMInput')
    LSTMSpectral = layers.LSTM(128,  return_sequences=True, activation='relu', name='LSTM1')(LSTMInput)
    LSTMSpectral = layers.LSTM(128, return_sequences=True, activation='relu', name='LSTM2')(LSTMSpectral)
    LSTMDense = layers.LSTM(128, return_sequences=False, activation='relu', name='LSTM3')(LSTMSpectral)
    LSTMDense = layers.Dense(128,  activation='relu', name='LSTMDense')(LSTMDense)

    LSTMSOFTMAX   = layers.Dense(nb_classes, activation='softmax', name='LSTMSOFTMAX')(LSTMDense)

    model = Model(input=[LSTMInput], output=[LSTMSOFTMAX])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 网络可视化
    plot_model(model, to_file='LSTM.png')
    model.summary()
    return model

def CNN(nb_classes, nb_features, img_rows, img_cols,time_step):

    CNNTInput = layers.Input(shape=[img_rows, img_cols, nb_features, time_step], name='CNNTInput')
    CONV3DT_1 = layers.Conv3D(64, (1, 1, 4), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3T_1')(CNNTInput)
    CONV3DT_2 = layers.Conv3D(64, (3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3T_2')(CONV3DT_1)
    Fusion3D_1 = layers.Concatenate()([CONV3DT_1, CONV3DT_2])
    Fusion3D_1 = layers.BatchNormalization(axis=-1)(Fusion3D_1)

    CNNSInput = layers.Input(shape=[img_rows, img_cols, time_step, nb_features], name='CNNSInput')
    CONV3DS_1 = layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3S_1')(CNNSInput)
    CONV3DS_2 = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3S_2')(CONV3DS_1)
    Fusion3D_2 = layers.Concatenate()([CONV3DS_1, CONV3DS_2])
    Fusion3D_2 = layers.BatchNormalization(axis=-1)(Fusion3D_2)

    Fusion3D_1 = layers.Reshape((int(Fusion3D_1.shape[1]), int(Fusion3D_1.shape[2]), int(Fusion3D_1.shape[3] * Fusion3D_1.shape[4])))(Fusion3D_1)
    Fusion3D_2 = layers.Reshape((int(Fusion3D_2.shape[1]), int(Fusion3D_2.shape[2]), int(Fusion3D_2.shape[3] * Fusion3D_2.shape[4])))(Fusion3D_2)

    Fusion2D_1 = layers.Concatenate()([Fusion3D_1, Fusion3D_2])
    Fusion2D_1 = layers.BatchNormalization(axis=-1)(Fusion2D_1)

    # CNNTInput = layers.Input(shape=[img_rows, img_cols, nb_features, time_step], name='CNNTInput')
    # CONV3DT_1 = layers.Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3T_1')(CNNTInput)
    # CONV3DT_2 = layers.Conv3D(64, (3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3T_2')(CONV3DT_1)
    #
    # CNNSInput = layers.Input(shape=[img_rows, img_cols, time_step, nb_features], name='CNNSInput')
    # CONV3DS_1 = layers.Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3S_1')(CNNSInput)
    # CONV3DS_2 = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name='CONV3S_2')(CONV3DS_1)
    #
    # CONV3DT_2 = layers.Reshape((int(CONV3DT_2.shape[1]), int(CONV3DT_2.shape[2]), int(CONV3DT_2.shape[3] * CONV3DT_2.shape[4])))(CONV3DT_2)
    # CONV3DS_2 = layers.Reshape((int(CONV3DS_2.shape[1]), int(CONV3DS_2.shape[2]), int(CONV3DS_2.shape[3] * CONV3DS_2.shape[4])))(CONV3DS_2)
    #
    # Fusion2D_1 = layers.Concatenate()([CONV3DT_2, CONV3DS_2])
    # Fusion2D_1 = layers.BatchNormalization(axis=-1)(Fusion2D_1)

    CONV2D_1 = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='CONV2D_1')(Fusion2D_1)
    CONV2D_2 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='CONV2D_2')(CONV2D_1)
    Fusion2D_2 = layers.Concatenate()([CONV2D_1, CONV2D_2])
    Fusion2D_2 = layers.BatchNormalization(axis=-1)(Fusion2D_2)
    CNNDense = layers.Flatten(name='FLATTEN5')(Fusion2D_2)

    CNNSOFTMAX = layers.Dense(nb_classes, activation='softmax', name='CNNSOFTMAX')(CNNDense)

    model = Model(input=[CNNTInput, CNNSInput], output=[CNNSOFTMAX])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 网络可视化
    plot_model(model, to_file='CACNN.png')
    model.summary()
    return model