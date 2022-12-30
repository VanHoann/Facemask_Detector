from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms

img_size = 224
batch_size = 16
id2label = {"0": "correct", "1": "incorrect", "2": "not wearing"}

# prepocess data to input to EfficientNet
transform = transforms.Compose(
    [transforms.Resize((img_size,img_size)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

def generate_df(image_dir, label):
    image_dir = [str(dir) for dir in image_dir]
    df = pd.DataFrame({"Filepath": image_dir, "Label": [label]*len(image_dir)})
    return df

class FaceMaskDataset(Dataset):
    def __init__(self, dir_label_df, transform=None):
        self.dir_label_df = dir_label_df
        self.transform = transform

    def __len__(self):
        return self.dir_label_df.shape[0]

    def __getitem__(self, idx):
        img_path = self.dir_label_df["Filepath"][idx]
        image = read_image(img_path, mode=ImageReadMode.RGB)/255
        label = torch.as_tensor([int(self.dir_label_df["Label"][idx])])
        if self.transform:
            image = self.transform(image.type(torch.DoubleTensor))
        return image, label

def prepare_data(data_dir="", verbose=False):
    # 3 dir to 3 classes
    with_mask_dir = data_dir+"/with_mask"
    incorrect_mask_dir = data_dir+"/incorrect_mask"
    without_mask_dir = data_dir+"/without_mask"

    # each class 3 mode: train, valid, test
    for dir in ["with_mask_dir", "incorrect_mask_dir", "without_mask_dir"]:
        exec(f'''{dir}_train = list(Path({dir}+"/train").glob(r'*jpg'))''')
        exec(f'''{dir}_valid = list(Path({dir}+"/valid").glob(r'*jpg'))''')
        exec(f'''{dir}_test = list(Path({dir}+"/test").glob(r'*jpg'))''')
        if verbose:
            exec(f'''print("{dir}", "train:", len({dir}_train), "valid:", len({dir}_valid), "test:", len({dir}_test))''')

    # merge each train, valid, test across 3 classes
    for mode in ["train", "valid", "test"]:
        exec(f'''with_mask_df = generate_df(with_mask_dir_{mode}, label=0)''')
        exec(f'''incorrect_mask_df = generate_df(incorrect_mask_dir_{mode}, label=1)''')
        exec(f'''without_mask_df = generate_df(without_mask_dir_{mode}, label=2)''')
        exec(f'''{mode}_df = pd.concat([with_mask_df, incorrect_mask_df, without_mask_df], axis=0, ignore_index=True)''')
        if verbose:
            exec(f'''print("{mode}:", {mode}_df.shape)''')
            exec(f'''print({mode}_df[:1])''')

    # prepare input format
    train_data = FaceMaskDataset(locals()["train_df"], transform=transform)
    valid_data = FaceMaskDataset(locals()["valid_df"], transform=transform)
    test_data = FaceMaskDataset(locals()["test_df"], transform=transform)

    if verbose:
        print("Dataset class:", len(train_data), len(valid_data), len(test_data), train_data[0][0].shape, train_data[0][1], float(train_data[0][0].max()), float(train_data[0][0].min()))

    # DataLoader    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
