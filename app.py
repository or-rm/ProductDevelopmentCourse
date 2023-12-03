from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from keras.models import load_model
import cv2
from PIL import Image #use to resize the image
import numpy as np

SIZE = 150
model = load_model('malaria-cnn-v1.keras')

app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
 
app.config['UPLOAD'] = upload_folder
 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        prediction = evaluateImage(filename)
        print(prediction)
        return render_template('image_render.html', img=img, prediction=prediction)
    return render_template('image_render.html')
 


def evaluateImage(filename) :
    image = cv2.imread(f'static/uploads/{filename}')
    image = Image.fromarray(image, "RGB")
    image = image.resize((SIZE, SIZE))
    input_img = np.expand_dims(image, axis=0)
    prediction =  model.predict(input_img)
    isAnMalaria = prediction[0][0] > 0.5
    
    return 'Tiene Malaria' if isAnMalaria else 'No tiene Malaria'
 
if __name__ == '__main__':
    app.run(debug=True, port=8001)