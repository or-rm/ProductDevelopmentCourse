from keras.models import load_model
import cv2
from PIL import Image #use to resize the image
import numpy as np

SIZE = 150


image = cv2.imread('static/uploads/01.png')
image = Image.fromarray(image, "RGB")
image = image.resize((SIZE, SIZE))


input_img = np.expand_dims(image, axis=0)


model = load_model('malaria-cnn-v1.keras')

prediction =  model.predict(input_img)

print(prediction)

