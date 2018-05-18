from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", required=True, help="saved full-model with structure and weights")

args = vars(ap.parse_args())

model = load_model(args["model"])

test_datagen = ImageDataGenerator(rescale=1. / 255)

img_width, img_height = 150, 150
nb_test_samples = 400
batch_size = 16

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

score = model.evaluate_generator(test_generator, nb_test_samples//batch_size, workers = 12)

print("Loss: ", score[0], "Accuracy: ", score[1])

# python classifier_with_keras_test.py --model "cat_and_dog_full_model.h5"
# python classifier_with_keras_test.py --model "classifier_with_VGG16_find_tune_full_model.h5"
