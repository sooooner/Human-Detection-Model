import tensorflow as tf
import matplotlib.pyplot as plt
import json
import requests
import pymysql
import pandas as pd
import base64
import cv2
from utils.utils import anchor_to_coordinate

def main():
    conn = pymysql.connect(host='mysql', user='root', charset='utf8') 
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    sql = "USE dacon;" 
    cursor.execute(sql) 

    sql = '''SELECT image_id, image FROM images'''
    cursor.execute(sql) 
    result = cursor.fetchall()
    df = pd.json_normalize(result)

    img = base64.decodebytes(df.iloc[0]['image'])
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
    plt.savefig('./test.png', dpi=200)

if __name__ == '__main__':
    main()

    