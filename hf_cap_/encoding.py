
from torch.utils.data import Dataset

class ImageCaptioningDataset(Dataset):
    def __init__ (self, dataset, processor): 
        self.dataset = dataset 
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images = item["image"], text = item["text"], padding = "max_length", return_tensors = "pt") # 데이터가 PyTorch의 텐서 형식으로 변환 

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()} # encoding 딕셔너리 각 항목에 대해 value 텐서 압축, 차원중 크기가 1인 차원이 있다면 차원 제거 

        return encoding
