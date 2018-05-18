from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import numpy as np

test_datagen = ImageDataGenerator(rescale=1. / 255)

img_width, img_height = 150, 150
nb_test_samples = 400
batch_size = 16

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

model_VGG16 = applications.VGG16(include_top=False, weights='imagenet')

test_data = model_VGG16.predict_generator(test_generator, nb_test_samples // batch_size)
test_labels = np.array([0] * (nb_test_samples // 2) + [1] * (nb_test_samples // 2))

model = load_model('classify_with_VGG16_bottleneck_full_model.h5')


score = model.test_on_batch(test_data, test_labels)

print("Loss: ", score[0], "Accuracy: ", score[1])