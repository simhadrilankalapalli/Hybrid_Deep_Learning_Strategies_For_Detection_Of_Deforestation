from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=0.2,
    fill_mode='nearest')

img=load_img('D:/Fathima/Python/deforestation/dataset/forest/35.jpg')
x=img_to_array(img)
x=x.reshape((1,)+ x.shape)


11253
11336

i=0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='D:/Fathima/Python/deforestation/dataset/forest',save_prefix='IMG', save_format='jpg'):
    i+=1
    if i>30:
        break