from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Dense

# args -------------------------------------------------------------------------------------------------
epochs = 9
batch_size = 100
image_width, image_height = 120, 120
# args -------------------------------------------------------------------------------------------------
# const ------------------------------------------------------------------------------------------------
WORK_DIR = './'
TRAIN_DIR = WORK_DIR + 'training_set/'
TEST_DIR = WORK_DIR + 'test_set/'


# const ------------------------------------------------------------------------------------------------
# main -------------------------------------------------------------------------------------------------
def main():
    print('creating generators ...')
    train_generator = create_generator(TRAIN_DIR)
    test_generator = create_generator(TEST_DIR)

    print('creating model ...')
    model = create_model()

    print('fitting data ...')
    fit(train_generator, test_generator, model)


# main -------------------------------------------------------------------------------------------------
# fit --------------------------------------------------------------------------------------------------
def fit(train_generator, test_generator, model):
    model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=100)
    accuracy = model.evaluate_generator(test_generator, steps=60)
    print("Accuracy: ", accuracy[1])


# fit --------------------------------------------------------------------------------------------------
# createBatches ----------------------------------------------------------------------------------------
def create_generator(directory_name):
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(
        directory_name,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary'
    )


# createBatches ----------------------------------------------------------------------------------------
# model ------------------------------------------------------------------------------------------------
def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# model ------------------------------------------------------------------------------------------------

main()
