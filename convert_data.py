import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    height_paths = glob(os.path.join('data','test_original','*.tif')) #3D스캐너로 레이저스캔한 높이 값
    height_paths = [height_path for height_path in height_paths if "Height" in height_path]
    
    #height_path = height_paths[0] #한장만 불러옴

    for height_path in height_paths:

        overlay_image = get_overlay_image(height_path)

        cv2.imshow('image', overlay_image)
        cv2.waitKey()


def get_overlay_image(height_path):


    luminance_path = height_path.replace('_Height', '_luminance') #스캐너 안의 카메라이미지

    height_image = Image.open(height_path)

    height_image = np.array(height_image) #numpy array로 바꿈
    height_image = (height_image-32768.0)*1.6 #제조사 제조값 (um)으로 변환식

    height, width = height_image.shape
    target_size = (int((width*0.4)), int(height*0.4)) #image size : w*h, 0.4배로 줄이기

    height_image = cv2.resize(height_image, target_size)

    mid_point = np.median(height_image)
    lsl_point = mid_point - 800 #휴리스틱 값
    usl_point = mid_point + 1500 #휴리스틱 값

    under_lsl_index = np.where(height_image < lsl_point) #h,w ->([h array값들], [w array값들])
    upper_usl_index = np.where(height_image > usl_point)

    #nomalization해서 0~1로 만들기
    normalized_data = (height_image-lsl_point) / (usl_point-lsl_point)
    normalized_data = (normalized_data*255).astype(np.uint8)

    normalized_data[under_lsl_index] = 0 #usl 초과, lsl 미만인 것들을 255 및 0으로 취급
    normalized_data[upper_usl_index] = 255

    luminance_image = Image.open(luminance_path)
    luminance_image = np.array(luminance_image)
    luminance_image = cv2.resize(luminance_image, target_size)

    luminance_image = (luminance_image/1024*255).astype(np.uint8) #1024로 나눔 : 0~1로 nolm. 그 후 *255

    #이미지 컬러화
    height_image_color = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET) #RGB
    luminance_image_color = cv2.cvtColor(luminance_image, cv2.COLOR_GRAY2BGR)

    overlay_image = cv2.addWeighted(height_image_color, 0.5, luminance_image_color, 0.5, 0.0) #마지막은 오프셋

    return overlay_image

# plt.hist(height_image)
# plt.show()

if __name__ == "__main__":
    main()