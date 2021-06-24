import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from werkzeug.utils import secure_filename
from utils.utils import anchor_to_coordinate
from flask import Flask, render_template, request

import cv2
import json
import pymysql
import requests
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def db_uploader(file_name):
    creat_test_img_sql = '''
    CREATE TABLE IF NOT EXISTS test_img (
    id INT(11) NOT NULL AUTO_INCREMENT,
    file_name VARCHAR(30) NOT NULL,
    PRIMARY KEY (id)
    );
    ''' 
    conn = pymysql.connect(host='mysql', user='root', charset='utf8') 
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    sql = "USE dacon;" 
    cursor.execute(sql) 
    cursor.execute(creat_test_img_sql) 

    sql = f"INSERT INTO test_img (file_name) VALUES ('{file_name}');"
    cursor.execute(sql)

def object_detection(file_name):
    image_path = './static/uploads/'
    img = tf.io.read_file(image_path + file_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.expand_dims(tf.image.resize(img, [432, 768])/255, 0)

    data = json.dumps({"signature_name": "serving_default", "instances": img.numpy().tolist()})

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://host.docker.internal:8501/v1/models/frcnn:predict', data=data, headers=headers)

    predictions = json.loads(json_response.text)['predictions']

    max_output_size = 3
    img_ = img[0].numpy().copy()

    scores_order = tf.argsort(predictions[0]['output_1'], direction='DESCENDING', axis=0)
    boxes = tf.squeeze(tf.gather(predictions[0]['output_2'], scores_order))
    boxes = boxes[boxes[:, 2] > 16]
    boxes = boxes[boxes[:, 3] > 16][:max_output_size]
    boxes = tf.math.reduce_mean(boxes, axis=0)

    anchor = anchor_to_coordinate(boxes.numpy())
    cv2.rectangle(
        img_, 
        (int(anchor[0]), int(anchor[2])), (int(anchor[1]), int(anchor[3])), 
        (1, 0, 0), 
        thickness=1
    )

    plt.imshow(img_)
    plt.axis('off')
    plt.savefig(f'./static/detect/{file_name}', dpi=200)

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/loading_page", methods = ['GET', 'POST'])
def loading():
    if request.method == 'POST':
        global file_name
        f = request.files['file1']
        file_name = secure_filename(f.filename)
        db_uploader(file_name)
        f.save(f'static/uploads/{file_name}')
        return render_template("loading_page.html")

@app.route('/file_uploaded', methods = ['GET', 'POST'])
def upload_file():
    # if request.method == 'POST':
    global file_name
    f = file_name
    object_detection(f)
    return render_template("image.html", img=f)

if __name__ == '__main__':
    app.run(debug=True)

