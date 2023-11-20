from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.layers import MaxPooling2D
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        # base_model = ResNet50(weights='imagenet')
        # self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
       self.model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
       self.model.trainable=False
       self.model= tf.keras.Sequential([
       self.model,
        MaxPooling2D()
        ])
       self.model.summary()

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x).flatten()  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize