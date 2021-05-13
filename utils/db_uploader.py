import argparse
import pandas as pd
import pymysql
from PIL import Image
import base64
from io import BytesIO

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=False, type=str, help="The path containing the data")
ap.add_argument("--host", required=True, type=str, help="Host of the MySQL server")
ap.add_argument("--user", required=True, type=str, help="User of the MySQL server")
args = ap.parse_args()

path = './res/'
if args.path:
    path = args.path

creat_keypoint_sql = '''
CREATE TABLE IF NOT EXISTS keypoint(
id INT(11) NOT NULL AUTO_INCREMENT,
nose_x FLOAT(11) NOT NULL,
nose_y FLOAT(11) NOT NULL,
left_eye_x FLOAT(11) NOT NULL,
left_eye_y FLOAT(11) NOT NULL,
right_eye_x FLOAT(11) NOT NULL,
right_eye_y FLOAT(11) NOT NULL,
left_ear_x FLOAT(11) NOT NULL,
left_ear_y FLOAT(11) NOT NULL,
right_ear_x FLOAT(11) NOT NULL,
right_ear_y FLOAT(11) NOT NULL,
left_shoulder_x FLOAT(11) NOT NULL,
left_shoulder_y FLOAT(11) NOT NULL,
right_shoulder_x FLOAT(11) NOT NULL,
right_shoulder_y FLOAT(11) NOT NULL,
left_elbow_x FLOAT(11) NOT NULL,
left_elbow_y FLOAT(11) NOT NULL,
right_elbow_x FLOAT(11) NOT NULL,
right_elbow_y FLOAT(11) NOT NULL,
left_wrist_x FLOAT(11) NOT NULL,
left_wrist_y FLOAT(11) NOT NULL,
right_wrist_x FLOAT(11) NOT NULL,
right_wrist_y FLOAT(11) NOT NULL,
left_hip_x FLOAT(11) NOT NULL,
left_hip_y FLOAT(11) NOT NULL,
right_hip_x FLOAT(11) NOT NULL,
right_hip_y FLOAT(11) NOT NULL,
left_knee_x FLOAT(11) NOT NULL,
left_knee_y FLOAT(11) NOT NULL,
right_knee_x FLOAT(11) NOT NULL,
right_knee_y FLOAT(11) NOT NULL,
left_ankle_x FLOAT(11) NOT NULL,
left_ankle_y FLOAT(11) NOT NULL,
right_ankle_x FLOAT(11) NOT NULL,
right_ankle_y FLOAT(11) NOT NULL,
neck_x FLOAT(11) NOT NULL,
neck_y FLOAT(11) NOT NULL,
left_palm_x FLOAT(11) NOT NULL,
left_palm_y FLOAT(11) NOT NULL,
right_palm_x FLOAT(11) NOT NULL,
right_palm_y FLOAT(11) NOT NULL,
spine2_back_x FLOAT(11) NOT NULL,
spine2_back_y FLOAT(11) NOT NULL,
spine1_waist_x FLOAT(11) NOT NULL,
spine1_waist_y FLOAT(11) NOT NULL,
left_instep_x FLOAT(11) NOT NULL,
left_instep_y FLOAT(11) NOT NULL,
right_instep_x FLOAT(11) NOT NULL,
right_instep_y FLOAT(11) NOT NULL,
PRIMARY KEY (id)
)
'''

creat_images_sql = '''
CREATE TABLE IF NOT EXISTS images (
id INT(11) NOT NULL AUTO_INCREMENT,
image_id INT(11) NOT NULL,
file_name VARCHAR(30) NOT NULL,
image MEDIUMBLOB NOT NULL,
PRIMARY KEY (id),
FOREIGN KEY (image_id) REFERENCES keypoint(id)
)
'''

def main(path, host, user):
    train_image_path = path + './train_imgs/'
    train = pd.read_csv(path + './train_df.csv')

    conn = pymysql.connect(host=host, user=user, charset='utf8') 
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    sql = "CREATE DATABASE IF NOT EXISTS dacon;" 
    cursor.execute(sql) 

    sql = "USE dacon;" 
    cursor.execute(sql) 
    cursor.execute(creat_keypoint_sql) 
    cursor.execute(creat_images_sql) 

    buffer = BytesIO()
    cols = 'nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y, left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y, left_wrist_x, left_wrist_y,right_wrist_x, right_wrist_y, left_hip_x, left_hip_y, right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y,left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, neck_x, neck_y, left_palm_x, left_palm_y, right_palm_x, right_palm_y, spine2_back_x, spine2_back_y, spine1_waist_x, spine1_waist_y, left_instep_x, left_instep_y, right_instep_x, right_instep_y'
    for i in range(len(train)):
        sql = f"INSERT INTO keypoint ({cols}) VALUES ({', '.join(list(map(str, train.iloc[i].values[1:])))})"
        cursor.execute(sql) 
        
        file_name = train.iloc[i]["image"]
        im = Image.open(train_image_path + file_name)
        im.save(buffer, format='jpeg')
        img_str = base64.b64encode(buffer.getvalue()).decode('UTF-8')
        sql = f"INSERT INTO images (image_id, file_name, image) VALUES {i+1, file_name, img_str}"
        cursor.execute(sql) 

    conn.commit()

if __name__ == '__main__':
    main(path, args.host, args.user)

