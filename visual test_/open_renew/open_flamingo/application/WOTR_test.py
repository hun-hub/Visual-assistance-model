import sys
import torch
from PIL import Image
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import os
import time
import glob

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
from PIL import Image
import os
import time

# 이미지 파일 경로
image_folder_path = "D:/visual_test -2"
start_index = 0
end_index = 9
record_file_path = "D:/visual_test -2/test2.txt"

# 생성된 텍스트를 저장할 리스트
texts_to_write = []
total_elapsed_time = 0  # 전체 경과 시간을 누적할 변수

# 이미지 순회image_paths = glob.glob(os.path.join(image_folder_path, "*.jpg"))
image_paths = glob.glob(os.path.join(image_folder_path, "*.jpg"))
# 각 이미지 경로에 대해 슬래시로 대체하여 수정
image_paths = [path.replace("\\", "/") for path in image_paths]

# 이미지 파일 경로 생성
for image_path in image_paths:
    # 이미지 전처리
    vision_x = preprocess_image(image_path)

    # 초기 텍스트 설정
    initial_text = "<image>An image of"

    # 텍스트 생성 시간 측정 시작
    start_time = time.time()

    # 텍스트 생성
    lang_x = tokenizer([initial_text], return_tensors="pt")
    generated_text = generate_text(vision_x, lang_x)
    generated_text = generated_text.replace("<image>", "").replace("<|endofchunk|>", "").strip()

    # 텍스트 생성 시간 측정 종료
    end_time = time.time()

    # 생성된 텍스트 출력
    print(f"Generated text for {image_path}: {generated_text}")

    # 캡션 생성에 소요된 시간 계산
    elapsed_time = end_time - start_time
    total_elapsed_time += elapsed_time

    # 시간 출력
    print(f"Caption generation time for {image_path}: {elapsed_time} seconds")

    # 생성된 텍스트 및 캡션 생성 시간을 리스트에 추가
    texts_to_write.append(f"{generated_text}\n")
    # texts_to_write.append(f"Caption generation time for {image_path}: {elapsed_time} seconds\n")

# 전체 이미지에 대한 평균 시간 계산
average_elapsed_time = total_elapsed_time / len(image_paths)

# 파일에 생성된 텍스트 쓰기
with open("generated_texts.txt", "w") as f:
    for text in texts_to_write:
        f.write(text)

# 평균 시간 출력
print(f"Average caption generation time: {average_elapsed_time} seconds")
# 전체 이미지에 대한 평균 시간 계산
average_elapsed_time = total_elapsed_time / (end_index - start_index + 1)

# 평균 시간을 파일에 기록
# texts_to_write.append(f"Average caption generation time: {average_elapsed_time} seconds\n")

# 파일 열기 (기존 파일에 덮어쓰기 모드로 열기)
with open(record_file_path, "w") as record_file:
    # 생성된 텍스트를 파일에 쓰기
    record_file.writelines(texts_to_write)

print("Texts recorded in", record_file_path)
print("Average caption generation time:", average_elapsed_time, "seconds")