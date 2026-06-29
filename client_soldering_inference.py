import requests
import json 
import time 
import os 
import cv2
from glob import glob 
import numpy as np
from shutil import copyfile

def main():
    retrain_dir = "need_retrain"
    os.makedirs(retrain_dir, exist_ok=True)

    image_dir = os.path.join("dataset", "test")
    val_images = glob(os.path.join(image_dir, '*.JPG'))

    idx = 0
    total_time = []
    while True:
        image_path = val_images[idx]
        
        begin_time = time.time() 
        image = cv2.imread(image_path)
        data = image.flatten().tobytes() 
        res = requests.post('http://127.0.0.1:5000/predict', data=data, timeout=10)
        if res.status_code == 200:
            instances = json.loads(res.text)
            image = draw_mask_on_image(instances, image)  # 불량 위치를 이미지에 표시
            image = ruled_inspection(instances, image)    # PASS/FAIL 룰베이스 판정
                        
        cv2.imshow('output', image)
        elapsed_time = time.time()-begin_time
        total_time.append(elapsed_time)
        print(idx, len(val_images)-1, os.path.basename(image_path), '{:2.3f}'.format(elapsed_time), '[s]')
        key = cv2.waitKey()
        if key == ord('a'):
            idx -= 1
        elif key == ord('d'):
            idx += 1
        elif key == ord('q'):
            break
        elif key == ord('s'):
            result_dir = "result"
            os.makedirs(result_dir, exist_ok=True)
            cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path)), image)
        elif key == ord('k'):
            idx += 1
        elif key == ord('n'):
            idx += 1
            print("AI Failed! Copy Data to ", retrain_dir)
            copyfile(image_path, os.path.join(retrain_dir, os.path.basename(image_path)))
        if idx < 0:
            idx = 0
        elif idx > len(val_images)-1:
            print("End of the data. Press q to quit.")
            idx = len(val_images)-1
            # break
    print("Average Elapsed Time:", np.mean(total_time))

def draw_mask_on_image(instances, image):

    colors = {'ok': (153,148,47),
              'short': (79,219,247), 
              'insufficient': (196,245,225),
              'no_solder':    (255, 0, 0),
              'solder_ball': (80,78,255)
              }
     
    for key in instances.keys(): #keys 호출
        for polygon in instances[key]:
            polygon = np.array(polygon).reshape(-1,2)

            image = cv2.drawContours(image,[polygon],-1,color=colors[key], thickness=2)
            image = cv2.putText(image,key,polygon[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colors[key],thickness=1) #소수점 둘째자리까지

    return image

def ruled_inspection(instances,image):

    short_lists = instances['short']
    insufficient_lists = instances['insufficient']
    no_solder_lists = instances['no_solder']
    solder_ball_lists = instances['solder_ball']

    if short_lists or insufficient_lists or no_solder_lists or solder_ball_lists :
        result = 'NG'
        color = (0,0,255)
    else:
        result = 'OK'
        color = (255,0,0)

    image = cv2.putText(image, result, (210,256), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=color, thickness=3)

    return image


if __name__ == "__main__":
    main()