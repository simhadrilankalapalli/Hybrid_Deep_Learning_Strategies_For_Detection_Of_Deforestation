from keras.models import Sequential, load_model
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from matplotlib import pyplot as plt

classifier = Sequential()

# 1st model

classifier.add(Convolution2D(512, (3, 3), input_shape=(64, 64, 3), activation='relu',strides=(3,3)))

classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Convolution2D(256, (2, 2), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(1, 1)))

classifier.add(Flatten())

classifier.add(Dense(units=64, activation='relu'))

classifier.add(Dense(units=2, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('D:\\Fathima\\Python\\deforestation\\dataset\\train',
                                                 target_size=(64,64),
                                                 batch_size=12,
                                                 class_mode='categorical'
                                                 )

test_set = test_datagen.flow_from_directory('D:\\Fathima\\Python\\deforestation\\dataset\\test',
                                            target_size=(64,64),
                                            batch_size=12,
                                            class_mode='categorical'
                                            )
print("\n Testing the data.....\n")

history=classifier.fit_generator(training_set,steps_per_epoch =20,epochs = 100,validation_data = test_set,verbose = 1)

classifier.save(r"D:\Fathima\Python\deforestation\CNN_1.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='training accuracy',color='green')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"D:\Fathima\Python\deforestation\CNN.png")
plt.show()

