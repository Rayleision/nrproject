import tensorflow as tf
from keras.layers import Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.models import Model, Sequential


def space_to_depth_x2(x):
    import tensorflow as tf
    return tf.nn.space_to_depth(x,block_size=2)

def Network1():
    input = Input(shape=(64, 64, 3))

    layer1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block1_conv1', use_bias=False)(input)
    layer1 = BatchNormalization(name='norm_1')(layer1)
    layer1 = Activation("relu")(layer1)

    layer2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block1_conv2', use_bias=False)(layer1)
    layer2 = BatchNormalization(name='norm_2')(layer2)
    layer2 = Activation("relu")(layer2)

    layer3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block1_conv3', use_bias=False)(layer2)
    layer3 = BatchNormalization(name='norm_3')(layer3)
    layer3 = Activation("relu")(layer3)

    layer4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block1_conv4', use_bias=False)(layer3)
    layer4 = BatchNormalization(name='norm_4')(layer4)
    layer4 = Activation("relu")(layer4)

    layer5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block1_conv5', use_bias=False)(layer4)
    layer5 = BatchNormalization(name='norm_5')(layer5)
    layer5 = Activation("relu")(layer5)

    layer6 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer5)
    skip_connection_1 = layer6

    layer7 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block2_conv1', use_bias=False)(layer6)
    layer7 = BatchNormalization(name='norm_7')(layer7)
    layer7 = Activation("relu")(layer7)

    layer8 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv2', use_bias=False)(layer7)
    layer8 = BatchNormalization(name='norm_8')(layer8)
    layer8 = Activation("relu")(layer8)

    layer9 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block2_conv3', use_bias=False)(layer8)
    layer9 = BatchNormalization(name='norm_9')(layer9)
    layer9 = Activation("relu")(layer9)

    layer10 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block2_conv4', use_bias=False)(layer9)
    layer10 = BatchNormalization(name='norm_10')(layer10)
    layer10 = Activation("relu")(layer10)

    layer11 = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='block2_conv5', use_bias=False)(layer10)
    layer11 = BatchNormalization(name='norm_11')(layer11)
    layer11 = Activation("relu")(layer11)

    layer12 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer11)
    skip_connection_1 = Lambda(space_to_depth_x2)(skip_connection_1)
    layer13 = concatenate([skip_connection_1, layer12])
    skip_connection_2 = layer13

    layer14 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block3_conv1', use_bias=False)(layer13)
    layer14 = BatchNormalization(name='norm_14')(layer14)
    layer14 = Activation("relu")(layer14)

    layer15 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block3_conv2', use_bias=False)(layer14)
    layer15 = BatchNormalization(name='norm_15')(layer15)
    layer15 = Activation("relu")(layer15)

    layer16 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv3', use_bias=False)(layer15)
    layer16 = BatchNormalization(name='norm_16')(layer16)
    layer16 = Activation("relu")(layer16)

    layer17 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block3_conv4', use_bias=False)(layer16)
    layer17 = BatchNormalization(name='norm_17')(layer17)
    layer17 = Activation("relu")(layer17)

    layer18 = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='block3_conv5', use_bias=False)(layer17)
    layer18 = BatchNormalization(name='norm_18')(layer18)
    layer18 = Activation("relu")(layer18)

    layer19 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer18)
    skip_connection_2 = Lambda(space_to_depth_x2)(skip_connection_2)

    layer20 = concatenate([skip_connection_2, layer19])
    layer21 = Conv2D(200, (1, 1), name='block4_conv1', use_bias=False)(layer20)
    layer21 = BatchNormalization(name='norm_21')(layer21)

    layer22 = GlobalAveragePooling2D(data_format=None)(layer21)
    output = Activation('softmax')(layer22)

    model = Model(inputs=[input], outputs=[output])

    return model

def Network2():
    input = Input(shape=(64, 64, 3))

    layer1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block1_conv1', use_bias=False)(input)
    layer1 = BatchNormalization(name='norm_1')(layer1)
    layer1 = Activation("relu")(layer1)

    skip_connection_1 = layer1

    layer2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv1', use_bias=False)(layer1)
    layer2 = BatchNormalization(name='norm_2')(layer2)
    layer2 = Activation("relu")(layer2)

    layer3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv2', use_bias=False)(layer2)
    layer3 = BatchNormalization(name='norm_3')(layer3)
    layer3 = Activation("relu")(layer3)

    layer4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv3', use_bias=False)(layer3)
    layer4 = BatchNormalization(name='norm_4')(layer4)
    layer4 = Activation("relu")(layer4)

    layer5 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv4', use_bias=False)(layer4)
    layer5 = BatchNormalization(name='norm_5')(layer5)
    layer5 = Activation("relu")(layer5)

    skip_connection_1 = Lambda(space_to_depth_x2)(skip_connection_1)
    layer6 = concatenate([skip_connection_1, layer5])

    layer7 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer6)
    skip_connection_2 = layer7

    layer8 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv1', use_bias=False)(layer7)
    layer8 = BatchNormalization(name='norm_8')(layer8)
    layer8 = Activation("relu")(layer8)

    layer9 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv2', use_bias=False)(layer8)
    layer9 = BatchNormalization(name='norm_9')(layer9)
    layer9 = Activation("relu")(layer9)

    layer10 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv3', use_bias=False)(layer9)
    layer10 = BatchNormalization(name='norm_10')(layer10)
    layer10 = Activation("relu")(layer10)

    layer11 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv4', use_bias=False)(layer10)
    layer11 = BatchNormalization(name='norm_11')(layer11)
    layer11 = Activation("relu")(layer11)

    skip_connection_2 = Lambda(space_to_depth_x2)(skip_connection_2)
    layer12 = concatenate([skip_connection_2, layer11])
    layer13 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer12)
    skip_connection_3 = layer13

    layer14 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv1', use_bias=False)(layer13)
    layer14 = BatchNormalization(name='norm_14')(layer14)
    layer14 = Activation("relu")(layer14)

    layer15 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv2', use_bias=False)(layer14)
    layer15 = BatchNormalization(name='norm_15')(layer15)
    layer15 = Activation("relu")(layer15)

    layer16 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv3', use_bias=False)(layer15)
    layer16 = BatchNormalization(name='norm_16')(layer16)
    layer16 = Activation("relu")(layer16)

    layer17 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv4', use_bias=False)(layer16)
    layer17 = BatchNormalization(name='norm_17')(layer17)
    layer17 = Activation("relu")(layer17)

    skip_connection_3 = Lambda(space_to_depth_x2)(skip_connection_3)
    layer18 = concatenate([skip_connection_3, layer17])

    layer19 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer18)

    layer20 = Conv2D(200, (1, 1), name='block5_conv1', use_bias=False)(layer19)
    layer20 = BatchNormalization(name='norm_20')(layer20)

    layer21 = GlobalAveragePooling2D(data_format=None)(layer20)
    output = Activation('softmax')(layer21)

    return output