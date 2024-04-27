import sys

# open_flamingo 패키지의 위치를 sys.path에 추가
sys.path.append('C:/Users/USER/Downloads/open_renew')



from open_flamingo import create_model_and_transforms


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14", # 모델의 구조, 아키텍처를 지정
    clip_vision_encoder_pretrained="openai", # CLIP (Contrastive Language-Image Pre-training) 모델의 가중치
    
    
    # MPT (Multimodal Pretrained Transformer) 모델의 언어 인코더와 토크나이저 가중치
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",  
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b", 
    cross_attn_every_n_layers=1,
    # cache_dir="C:/Users/USER/Downloads/open_renew/open_flamingo/train"
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

# 이 부분이 진짜 가중치 가져오는 부분 
checkpoint_path = 'C:/Users/USER/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt' # C:/Users/USER/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt
model.load_state_dict(torch.load(checkpoint_path), strict=False)



from PIL import Image
import requests
import torch
import glob
import cv2
from video import process_video_frames 

video_path = "C:/Users/USER/Downloads/visual test/open_renew/video/video1.mp4"
process_video_frames(video_path)



def load_images_from_directory(directory):
    image_paths = glob.glob(directory + "/*.png")
    images = [Image.open(image_path) for image_path in image_paths]
    return images

# "img/shot" 디렉토리에서 이미지 가져오기
shot_images = load_images_from_directory("C:/Users/USER/Downloads/open_renew/img/shot")

# "img/query" 디렉토리에서 이미지 가져오기
query_image_path = "C:/Users/USER/Downloads/open_renew/img/query/img.png"
query_image = Image.open(query_image_path)




# 가져온 이미지를 이용하여 vision_x 리스트 구성
vision_x = []
for shot_image in shot_images:
    vision_x.append(image_processor(shot_image).unsqueeze(0))

vision_x.append(image_processor(query_image).unsqueeze(0)) 
vision_x = torch.cat(vision_x, dim=0) # cat 이용해서 하나의 텐서로 연결, 하나의 배치로 포함하기 위해  
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: 텍스트 전처리 
 <image> 이미지와 관련된 텍스트 시작점
 <endofchunk> 이 토큰은 이미지와 관련된 텍스트 부분의 끝
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
from stt import recognize_speech
result = recognize_speech()
print(result)





tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>{result}<|endofchunk|><image>An image of"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x, # 이미지 데이터가 포함된 텐서 
    lang_x=lang_x["input_ids"], # 토큰화된 텍스트의 텐서
    attention_mask=lang_x["attention_mask"], # 각 토큰의 중요도를 제어
    max_new_tokens=20, # 생성할 최대 토큰 수
    num_beams=3, # 생성할 후보 문장 수 
    repetition_penalty=2.0 
)

generated_text_decoded = tokenizer.decode(generated_text[0]).split("<image>")[-1]
cleaned_text = generated_text_decoded.replace("<|endofchunk|", "")
output = cleaned_text.rstrip("<image>").rstrip().strip()

print("Generated text: ", output)





import pyttsx3


engine = pyttsx3.init()

rate = engine.getProperty('rate')
engine.setProperty('rate', 150)


voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)


engine.say(output)
engine.runAndWait()



"""
engine.getProperty('voices')
engine.say(output)
engine.runAndWait()




if __name__ == "__main__":
    input_text = input("Enter the text to convert to speech: ")
    text_to_speech(input_text)
"""


import os 
# "img/shot" 디렉토리 내의 모든 파일 삭제
shot_directory = "C:/Users/USER/Downloads/open_renew/img/shot"
for filename in os.listdir(shot_directory):
    file_path = os.path.join(shot_directory, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")

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
