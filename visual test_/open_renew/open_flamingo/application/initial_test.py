import sys
import torch
from PIL import Image
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import os

# open_flamingo 패키지의 위치를 sys.path에 추가
sys.path.append('C:/Users/USER/Downloads/open_renew')

# create_model_and_transforms 함수를 사용하여 모델, 이미지 프로세서, 토크나이저를 만듭니다.
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

# Hugging Face Hub에서 모델 체크포인트를 다운로드합니다.
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

# 쿼리 이미지 경로를 설정합니다.


# 전처리된 이미지를 생성합니다.
def preprocess_image(image_path):
    query_image = Image.open(image_path)
    vision_x = [image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x

# 텍스트를 생성하는 함수를 정의합니다.
def generate_text(vision_x, lang_x, max_new_tokens=20, num_beams=3, repetition_penalty=2.0):
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty
    )
    return tokenizer.decode(generated_text[0])
# 이미지 파일 경로
image_folder_path = "D:/data_set/"
start_index = 1
end_index = 100
record_file_path = "D:/data_set/record.txt"

# 파일 열기 (기존 파일에 덮어쓰기 모드로 열기)
with open(record_file_path, "w") as record_file:
    # 이미지 순회
    for i in range(start_index, end_index + 1):
        # 이미지 파일 경로 생성
        image_path = os.path.join(image_folder_path, f"VizWiz_test_{i:08}.jpg")

        # 이미지 전처리
        vision_x = preprocess_image(image_path)

        # 초기 텍스트 설정
        initial_text = "<image>An image of"

        # 텍스트 생성
        lang_x = tokenizer([initial_text], return_tensors="pt")
        generated_text = generate_text(vision_x, lang_x)

        # 생성된 텍스트 출력
        print(f"Generated text for {image_path}: {generated_text}")

        # 생성된 텍스트를 파일에 쓰기
        record_file.write(f"Generated text for {image_path}: {generated_text}\n")

print("Texts recorded in", record_file_path)