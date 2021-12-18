from keras_preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
import pandas as pd
import matplotlib.pyplot as plt


def Image_augmentation(network_num, val_annotations='./tiny-imagenet-200/val/val_annotations.txt', train_data_address='./tiny-imagenet-200/train/',
                       val_data_address='./tiny-imagenet-200/val/images/'):

    ''' convert val data from txt to dataframe '''
    data_list = []
    with open(val_annotations,'r')as datafile:
        for line in datafile:
            if line.count('\n') == len(line):
                continue
            for kv in [line.strip().split('\t')]:
                data_list.append(kv[:2])
        data_dataframe = pd.DataFrame(data_list, columns=['Filename', 'Class'])

    ''' use imgaug for augmentation '''
    if network_num == 1:
        # image_aug = iaa.SomeOf((0, 5), [iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}), iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)),
        #                                    iaa.Affine(rotate=(-10, 10)), iaa.AdditiveGaussianNoise(scale=0.05 * 255), iaa.CropAndPad(percent=(-0.1, 0.1))])

        image_aug = iaa.SomeOf((0, None), [iaa.Affine(scale=(0.5, 1.5)), iaa.Affine(rotate=20),
    iaa.CoarseDropout((0.0, 0.2), size_percent=(0.05, 0.07)), iaa.AdditiveGaussianNoise(scale=0.05*255), iaa.CropAndPad(percent=(-0.25, 0.25))
     ])                                          
    elif network_num == 2:
        image_aug = iaa.SomeOf(10, [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.GaussianBlur(sigma=(0.0, 3.0)), iaa.CropAndPad(percent=(-0.1, 0.1)),
                                    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}), iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                                    iaa.Affine(rotate=(-10, 10)), iaa.Affine(shear=(-10, 10)), iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)), iaa.Multiply((0.5, 1.5))])

    ''' load data '''
    #train_datagen = ImageDataGenerator(preprocessing_function=image_aug.augment_image, rescale=1. / 255)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)


    train_generator = train_datagen.flow_from_directory(directory=train_data_address, target_size=(64, 64), batch_size=64, shuffle=True, seed=36)
    validation_generator = valid_datagen.flow_from_dataframe(data_dataframe, directory=val_data_address, x_col='Filename', y_col='Class', target_size=(64, 64), batch_size=64, shuffle=True, seed=36)
    # x_batch, y_batch = next(train_generator)

    return train_generator, validation_generator

