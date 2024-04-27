import sys 
sys.path.append('C:/Users/USER/Desktop/hf_cap')
import load_dataset 
from encoding import ImageCaptioningDataset
from transformers import AutoProcessor #  Hugging Face의 Transformers 라이브러리에서 AutoProcessor 클래스 import
from torch.utils.data import DataLoader 
from PIL import Image 
import numpy as np 


def main():
    
    
    dataset = load_dataset.preprocess_data('C:/Users/USER/Desktop/hf_cap/data/json', 'C:/Users/USER/Desktop/hf_cap/data/img/')
    processor = AutoProcessor.from_pretrained("microsoft/git-base") # 언어모델 
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = 2) # get batches of data from the dataset 
    batch = next(iter(train_dataloader))




    dataset = load_dataset.preprocess_data('C:/Users/USER/Desktop/hf_cap/valid/json', 'C:/Users/USER/Desktop/hf_cap/valid/img/')
    processor = AutoProcessor.from_pretrained("microsoft/git-base") # 언어모델 
    valid_dataset = ImageCaptioningDataset(dataset, processor)
    valid_dataloader = DataLoader(valid_dataset, shuffle = True, batch_size = 2) # get batches of data from the dataset 
    batch = next(iter(valid_dataloader))





    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    # 출력 레이어를 제외한 모든 파라미터들의 requires_grad를 False로 설정
    for name, param in model.named_parameters():
        if 'output' not in name:  # 출력 레이어를 제외한 파라미터들만 동결하지 않습니다.
            param.requires_grad = False


    outputs = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["input_ids"])
    outputs.loss



    import torch 


    optimizer  = torch.optim.AdamW(model.parameters(), lr = 5e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)



    if torch.cuda.is_available():
            print("fdfdf")
            device = torch.device('cuda', torch.cuda.current_device())
            torch.backends.cudnn.benchmark = True
    else:
            print("no gpu")
            device = torch.device('cpu')



    model.train()

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model.to(device)
    """

    

    model.to(device)

    # total_train_loss = 0.0
    for epoch in range(40):
        # Training Phase
        batch_loss = 0.0
        print("Epoch:", epoch)
        
        model.train()  # 모델을 학습 모드로 설정
        
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_epoch_loss = batch_loss / len(train_dataloader)
        print("Epoch Average Train Loss:", avg_epoch_loss)

        # Save the model weights after each epoch
        save_path = 'C:/Users/USER/Desktop/hf_cap/weights/model_weights_epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), save_path)

        # Validation Phase
        total_valid_loss = 0.0
        valid_batch_loss = 0.0
        model.eval()  # 모델을 평가 모드로 설정
        
        # Load the model weights for validation
        model.load_state_dict(torch.load(save_path))
        model.to(device)

        with torch.no_grad():  # 기울기 계산 비활성화
            for idx, batch in enumerate(valid_dataloader):
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)

                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=input_ids)

                loss = outputs.loss
                valid_batch_loss += loss.item()

            avg_epoch_valid_loss = valid_batch_loss / len(valid_dataloader)
            print("Epoch Average Validation Loss:", avg_epoch_valid_loss)









    from PIL import Image
    import json
    import os

    # 경로 설정

    # image_folder = "C:/Users/USER/Desktop/hf_cap/valid/img"
    # jsonl_file = "C:/Users/USER/Desktop/hf_cap/valid/json/pred_captions.jsonl"

    # 이 부분 eval 할 때 바꿔주면 됨 
    image_folder = "C:/Users/USER/Desktop/hf_cap/eval/img"
    jsonl_file = "C:/Users/USER/Desktop/hf_cap/eval/json/eval_pred_captions.jsonl"


    # 모델로 생성한 캡션 captions.jsonl에 작성하기 
    with open(jsonl_file, 'w') as f:
        # 이미지 파일 경로 읽어오기
        image_files = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]

        for image_file in image_files:
            # 이미지 로드
            if image_file.endswith(".jpg"):
                image = Image.open(image_file)
            
            # 이미지를 모델 입력 형식으로 변환
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values
            
            # 모델에 입력하여 캡션 생성
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 생성된 캡션 출력
            # print(f"file_name: {image_file}, text: {generated_caption}")

            # 캡션을 JSONL 파일에 작성
            file_name = os.path.basename(image_file)
            data = {"file_name": file_name, "text": generated_caption}
            f.write(json.dumps(data) + "\n")


    
   




if __name__ == '__main__':
    main()


