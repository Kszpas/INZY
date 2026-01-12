from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    'dataset/train',
    target_size=(64,64),
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    'dataset/val',
    target_size=(64,64),
    class_mode='categorical'
)

print("Train:", train_gen.class_indices)
print("Val:", val_gen.class_indices)