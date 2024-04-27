import json
import glob
from datasets import load_dataset
from torch.utils.data import Dataset

def preprocess_data(json_folder_path, root_folder_path):
    json_paths = glob.glob(json_folder_path + '/*.json')
    captions = []

    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                file_name = item["image_filename"]
                text = item["captions"][0]
                captions.append({"file_name": file_name, "text": text})

    with open(root_folder_path + "metadata.jsonl", 'w') as f:
        for item in captions:
            f.write(json.dumps(item) + "\n")

    dataset = load_dataset("imagefolder", data_dir=root_folder_path, split="train")
    return dataset