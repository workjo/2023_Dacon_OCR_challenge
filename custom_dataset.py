import random

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from straug.blur import GaussianBlur

from config import CFG

df = pd.read_csv('./train.csv')

# 제공된 학습데이터 중 1글자 샘플들의 단어사전이 학습/테스트 데이터의 모든 글자를 담고 있으므로 학습 데이터로 우선 배치
df['len'] = df['label'].str.len()
train_v1 = df[df['len'] == 1]

# 제공된 학습데이터 중 2글자 이상의 샘플들에 대해서 단어길이를 고려하여 Train (80%) / Validation (20%) 분할
df = df[df['len'] > 1]
train_v2, val, _, _ = train_test_split(df, df['len'], test_size=0.1, random_state=CFG['SEED'])  # origin_221229

# 학습 데이터로 우선 배치한 1글자 샘플들과 분할된 2글자 이상의 학습 샘플을 concat하여 최종 학습 데이터로 사용
train = pd.concat([train_v1, train_v2])

# 학습 데이터로부터 단어 사전(Vocabulary) 구축
train_gt = [gt for gt in train['label']]
train_gt = "".join(train_gt)
letters = sorted(list(set(list(train_gt))))
CFG['character'] = "".join(letters)

vocabulary = ["-"] + letters
idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
char2idx = {v: k for k, v in idx2char.items()}

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, CFG, train_mode=True):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.train_mode = train_mode
        self.CFG = CFG
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        image = Image.open(self.img_path_list[index]).convert('RGB')

        if self.train_mode:
            if self.CFG['aug']:
                aug_prob = 0.5  # 원하는 확률 선택
                if random.random() < aug_prob:
                    image = GaussianBlur()(image)

            image = self.train_transform(image, self.CFG)
        else:
            image = self.test_transform(image, self.CFG)

        if self.label_list is not None:
            text = self.label_list[index]
            return image, text
        else:
            return image

    # Image Augmentation
    def train_transform(self, image, CFG):
        if CFG['input_channel'] == 3:
            transform_ops = transforms.Compose([
                transforms.Resize((CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        elif CFG['input_channel'] == 1:
            transform_ops = transforms.Compose([
                transforms.Resize((CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE'])),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(mean=(0.5), std=(0.5))
            ])

        return transform_ops(image)

    def test_transform(self, image, CFG):
        if CFG['input_channel'] == 3:
            transform_ops = transforms.Compose([
                transforms.Resize((CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        elif CFG['input_channel'] == 1:
            transform_ops = transforms.Compose([
                transforms.Resize((CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE'])),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(mean=(0.5), std=(0.5))
            ])

        return transform_ops(image)
