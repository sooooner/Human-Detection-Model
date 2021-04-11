import numpy as np
import cv2
from math import pi, cos, sin

pixel_shifts = [25]
rotation_angles = [10]
inc_brightness_ratio = 1.2
dec_brightness_ratio = 0.8
noise_ratio = 0.008

def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=1)
    for idx, sample_keypoints in enumerate(keypoints):
        if idx%2 == 0:
            flipped_keypoints.append(768. -sample_keypoints)
        else:
            flipped_keypoints.append(sample_keypoints)
    
    for i in range(8):
        flipped_keypoints[2+(4*i):4+(4*i)], flipped_keypoints[4+(4*i):6+(4*i)] = flipped_keypoints[4+(4*i):6+(4*i)], flipped_keypoints[2+(4*i):4+(4*i)]
    flipped_keypoints[36:38], flipped_keypoints[38:40] = flipped_keypoints[38:40], flipped_keypoints[36:38]
    flipped_keypoints[44:46], flipped_keypoints[46:48] = flipped_keypoints[46:48], flipped_keypoints[44:46]
    
    return flipped_images, flipped_keypoints


def shift_images(images, keypoints):
    images = images.numpy()
    shifted_images = []
    shifted_keypoints = []
    for shift in pixel_shifts:   
        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
            M = np.float32([[1,0,shift_x],[0,1,shift_y]])
            shifted_keypoint = np.array([])
            shifted_x_list = np.array([])
            shifted_y_list = np.array([])
            shifted_image = cv2.warpAffine(images, M, (768, 432), flags=cv2.INTER_CUBIC)
            for idx, point in enumerate(keypoints):
                if idx%2 == 0: 
                    shifted_keypoint = np.append(shifted_keypoint, point+shift_x)
                    shifted_x_list = np.append(shifted_x_list, point+shift_x)
                else: 
                    shifted_keypoint =np.append(shifted_keypoint, point+shift_y)
                    shifted_y_list = np.append(shifted_y_list, point+shift_y)
            if np.all(0.0<shifted_x_list) and np.all(shifted_x_list<768) and np.all(0.0<shifted_y_list) and np.all(shifted_y_list<432):
                shifted_images.append(shifted_image.reshape(432, 768, 3))
                shifted_keypoints.append(shifted_keypoint)

    return shifted_images, shifted_keypoints

def rotate_augmentation(images, keypoints):
    images = images.numpy()
    rotated_images = []
    rotated_keypoints = []
    
    for angle in rotation_angles:
        for angle in [angle,-angle]:
            M = cv2.getRotationMatrix2D((768//2,432//2), angle, 1.0)
            M = np.array(M, dtype=np.float32)
            angle_rad = -angle*pi/180
            rotated_image = cv2.warpAffine(images, M, (768,432))
            rotated_images.append(rotated_image)
            
            rotated_keypoint = keypoints.copy()
            rotated_keypoint[0::2] = rotated_keypoint[0::2] - 768//2
            rotated_keypoint[1::2] = rotated_keypoint[1::2] - 432//2
            
            for idx in range(0,len(rotated_keypoint),2):
                rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)

            rotated_keypoint[0::2] = rotated_keypoint[0::2] + 768//2
            rotated_keypoint[1::2] = rotated_keypoint[1::2] + 432//2
            rotated_keypoints.append(rotated_keypoint)
        
    return rotated_images, rotated_keypoints

def alter_brightness(images):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*inc_brightness_ratio, 0.0, 1.0)
    dec_brightness_images = np.clip(images*dec_brightness_ratio, 0.0, 1.0)
    altered_brightness_images.append(inc_brightness_images)
    altered_brightness_images.append(dec_brightness_images)
    return altered_brightness_images

def add_noise(images):
    images = images.numpy()
    noise = noise_ratio * np.random.randn(432,768,3)
    noise = noise.astype(np.float32)
    noisy_image = cv2.add(images, noise)
    return noisy_image