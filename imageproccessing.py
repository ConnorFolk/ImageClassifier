# -*- coding: utf-8 -*-

import uuid
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions


from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

app= Flask(__name__)

model = ResNet50(weights='imagenet') #automically gives us a 50 layer resnet with image weights
photos = UploadSet(name= 'photos', extensions= IMAGES) # we just want to upload images so IMAGES is list of valid images


app.config['UPLOADED_PHOTOS_DEST'] = 'static/img' #configure a directory where we are actually going to be saving these images
configure_uploads(app, upload_sets=photos) # we upload a file and it automatically saves 

@app.route('/', methods= ['GET', 'POST']) # within html you can get a webpage (display), post is uploading something to
#so we need to support both get and post bc we are uploading an image
def upload():
    if request.method == 'POST' and 'photo' in request.files: # if we are doing a post
        filename= photos.save(request.files['photo'], name= uuid.uuid4().hex[:8] + '.')  #save file with unique identifier name for each new photo so you do not overwrite anything     
        return redirect(url_for('show', filename= filename)) #redirect user to next function to see result
    else: #if we are doing a get we give them the webpage
        return render_template('upload.html') #this calls the html file in templates folder that we named upload (this folder must always be called templates bc that is where the function looks)
        
@app.route('/photo/<filename>')
def show(filename):
    img_path = app.config['UPLOADED_PHOTOS_DEST'] + '/' + filename
    img= image.load_img(img_path, target_size=(224,224)) #resizes image so it can be used by model
    x= image.img_to_array(img) #convert image to array
    x= x[np.newaxis, ...] #give this a batch size of 1 (give another dimension at beginning)
    x= preprocess_input(x) #preprocessing for Resnet
    
    y_pred = model.predict(x) #get predictions
    predictions= decode_predictions(y_pred, top=5)[0] #put predictions in readible form 
    url= photos.url(filename)
    return render_template('view_results.html', filename=filename, url=url, predictions=predictions)
    
if __name__=="__main__":
    app.run(port=1001)


