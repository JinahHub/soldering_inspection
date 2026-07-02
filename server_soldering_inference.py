# app.py 파일 생성
from flask import Flask, request, jsonify
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from glob import glob
import platform
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.amp import GradScaler
from tools.engine import train_one_epoch, evaluate, test_one_epoch
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tools.utils import collate_fn
from torchvision import tv_tensors

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomRotation(degrees=(-5,5)))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.Resize((512,512)))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# Flask 애플리케이션 생성
app = Flask(__name__)

n_classes = 5

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

# load an instance segmentation model pre-trained on COCO
model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes+1) # num_classes + background

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256

# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, n_classes+1)
model = model.to(device)

state_dict = torch.load(os.path.join('output', 'model_best.pth'), weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

eval_transform = get_transform(train=False)

# 루트(/) 경로에 접속했을 때 실행될 함수 정의
@app.route('/predict', methods = ['POST'])

def predict():

    instances = {
            "ok": [],
            "short": [],
            "insufficient": [],
            "no_solder": [],
            "solder_ball": [],
            }

    data = request.data
    image_np = np.frombuffer(data,dtype=np.uint8) #데이터타입정해줌
    image_np = image_np.reshape(512,512,3)

    image = Image.fromarray(image_np[:,:,::-1]) #nparray -> PIL 바꿈, BGR -> RGB로 변환
    image = tv_tensors.Image(image) #image tensor로 변환 : transform 할 때 각각 변환 규칙에 맞게끔 변환함
    image = eval_transform(image) # train은 안함
    model.eval() #평가모드전환
    with torch.no_grad():
        predictions = model([image.to(device)])
    pred = predictions[0]
    if len(pred['labels']) > 0: #추론 실패 깃발
        mask_on_image, instances = draw_mask_on_image(image_np.copy(), pred, score_threshold=0.8)
        cv2.imwrite('mask_on_image.png',mask_on_image) #결과 이미지 로컬 저장
    return jsonify(instances)


def draw_mask_on_image(image, pred,score_threshold=0.8):

    h,w,c = image.shape

    masks = (pred["masks"] > 0.2) #true와 false로 이루어짐, instance의 pred의 threshold
    masks = masks.cpu().numpy().astype(np.uint8)*255 #shape([마스크수,1,h,w])

    mask_list = []
    for mask in masks:
        mask = np.squeeze(mask) #어디든 있는 1만 억제 : shape을 맞추기 위해
        mask = cv2.resize(mask, (w,h)) #사이즈 원복
        mask_list.append(mask) #mask가 mask_list로 쌓여감
    
    masks = np.array(mask_list)
    scores = pred['scores'].cpu().numpy() #score 텐서 -> numpy로
    labels = pred['labels'].cpu().numpy() #score크기 큰것부터 작은거순으로 뜸
    
    valid_indicies = np.where(scores > score_threshold)
    masks = masks[valid_indicies]
    scores = scores[valid_indicies]
    labels = labels[valid_indicies]

    colors = {1: (153,148,47),
              2: (79,219,247),
              3: (196,245,225),
              4: (58,145,252),
              5: (80,78,255)
              }

    ok = []
    short = []
    insufficient = []
    no_solder = []
    solder_ball = []

    for mask, score, label in zip(masks, scores, labels): #zip쓸려면 len 같아야함
        
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #마스크 윤곽선 따기 (이미지 위에 오버랩 했을 때 잘보일려고), ([ [x,y], [x,y] ...])
        image = cv2.drawContours(image, contours, -1, colors[label], 2) #윤곽선 좌표를 선으로 이어줌
        image = cv2.putText(image, f'{score:.2f}',contours[0][0][0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colors[label],thickness=1) #소수점 둘째자리까지
    
        polygon = contours[0].squeeze().flatten().tolist() #가독성을 위해서 flatten numpy -> list로 변환, contours = (array[..],assray[..]), 0번째면 numpy array
        if label==1 :
            ok.append(polygon)
        if label==2 :
            short.append(polygon)
        if label==3 :
            insufficient.append(polygon)
        if label==4 :
            no_solder.append(polygon)
        if label==5 :
            solder_ball.append(polygon)

    instances = {'ok' : [], #[[x,y,x,y...],[x,y,x,y...]]
                 'short' : [], #[[x,y,x,y...],[x,y,x,y...]]
                 'insufficient' : [], #[[x,y,x,y...],[x,y,x,y...]]
                 'no_solder' : [], #[[x,y,x,y...],[x,y,x,y...]]
                 'solder_ball' : []} #[[x,y,x,y...],[x,y,x,y...]]

    instances['ok'] = ok
    instances['short'] = short
    instances['insufficient'] = insufficient
    instances['no_solder'] = no_solder
    instances['solder_ball'] = solder_ball


    return image, instances



if __name__ == '__main__':
    # 서버 실행 (debug=True는 코드가 바뀔 때마다 서버 자동 재시작)
    app.run(host="127.0.0.1", debug=False, port=5000)
