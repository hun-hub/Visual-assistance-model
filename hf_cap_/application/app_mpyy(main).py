import sys 
sys.path.append('C:/Users/USER/Desktop/hf_cap')
import load_dataset 
from encoding import ImageCaptioningDataset
from transformers import AutoProcessor #  Hugging Face의 Transformers 라이브러리에서 AutoProcessor 클래스 import
from torch.utils.data import DataLoader 
from PIL import Image 
import numpy as np 


import torch 


from open_flamingo import create_model_and_transforms

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

processor = AutoProcessor.from_pretrained("microsoft/git-base") 
# 이 부분이 진짜 가중치 가져오는 부분 




if torch.cuda.is_available():
        print("fdfdf")
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
else:
        print("no gpu")
        device = torch.device('cpu')

model.to(device)





checkpoint_path = 'C:/Users/USER/Desktop/hf_cap/weights/model_weights_epoch_39.pth' # C:/Users/USER/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt
model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu')))
model.to(device)






from PIL import Image
import requests
import torch
import glob
import cv2
from video import process_video_frames 

video_path = "C:/Users/USER/Downloads/visual test/open_renew/video/video3.mp4"
process_video_frames(video_path)



# "img/query" 디렉토리에서 이미지 가져오기
query_image_path = "C:/Users/USER/Downloads/open_renew/img/query/img.png"
query_image = Image.open(query_image_path)





    
    # 이미지를 모델 입력 형식으로 변환
inputs = processor(images=query_image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

# 모델에 입력하여 캡션 생성
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"{generated_caption}")

import pyttsx3


engine = pyttsx3.init()

rate = engine.getProperty('rate')
engine.setProperty('rate', 150)


voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)


engine.say(generated_caption)
engine.runAndWait()



import os 



# "img/query" 디렉토리 내의 모든 파일 삭제
query_directory = "C:/Users/USER/Downloads/open_renew/img/query"
for filename in os.listdir(query_directory):
    file_path = os.path.join(query_directory, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")
