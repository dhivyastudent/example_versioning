import unittest
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.image import resize


class TestModel(unittest.TestCase):

    top_model_complete = "top_model_complete.h5"
    samples_path = "samples"

    def setUp(self):
        # load trained model
        self.restored_model = load_model(self.top_model_complete)

    def convert_img(self, img_path):
        # load image
        sample_image = load_img(img_path, target_size=(224, 224))
        sample_img = img_to_array(sample_image) / 255.0
        sample_img = np.expand_dims(sample_img, axis=0)

        # load base VGG16
        base_model = applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )

        # extract features
        layer_name = "block5_pool"
        output_layer = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(layer_name).output
        )

        converted_img = output_layer.predict(sample_img)

        # resize to match training input
        resized_sample = resize(converted_img, size=(4, 4))
        resized_sample = np.expand_dims(resized_sample[0], axis=0)

        return resized_sample.numpy()

    def test_sample1(self):
        sample1 = os.path.join(self.samples_path, "sample1.jpg")

        print("\n📸 Testing image:", sample1)

        resized = self.convert_img(sample1)

        result = self.restored_model.predict(resized)

        print("🔢 Raw prediction value:", result)

        if result[0][0] > 0.5:
            prediction = "dog"
        else:
            prediction = "cat"

        print("🐾 Final Prediction:", prediction)

        # ✅ Save output to file (important for GitHub Actions)
        with open("prediction.txt", "w") as f:
            f.write(f"Image: {sample1}\n")
            f.write(f"Raw prediction: {result}\n")
            f.write(f"Final prediction: {prediction}\n")

        # assertion
        self.assertEqual(prediction, "dog", "Predicted class is wrong")


if __name__ == "__main__":
    unittest.main()
