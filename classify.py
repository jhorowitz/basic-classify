from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

batch_size = 16
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    "training_data/train",
    target_size=(300, 300),
    batch_size=batch_size,
    class_mode='binary')

val_gen = val_datagen.flow_from_directory("training_data/validate")


def add_conv_set(layer, conv_size):
    layer = Conv2D(conv_size, (3, 3), activation='relu')(layer)
    layer = Conv2D(conv_size, (3, 3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(.25)(layer)
    return layer


inp = Input((300, 300, 3))

X = add_conv_set(inp, 32)
X = add_conv_set(X, 64)

X = Flatten()(X)
X = Dense(256, activation='relu')(X)
X = Dropout(.25)(X)
X = Dense(1, activation='sigmoid')(X)

model = Model(inputs=inp, outputs=X)

model.compile("rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_gen,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=val_gen,
    validation_steps=800 // batch_size)

model.save_weights('model_save.h5')
