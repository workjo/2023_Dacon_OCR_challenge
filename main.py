import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init

import utils
from custom_dataset import CustomDataset, train, val
import clova_model
import custom_train
import custom_test
from config import CFG, now

import warnings
warnings.filterwarnings(action='ignore')

# configuration 설정 및 시드 고정
####################################################################################
os.makedirs(f'./saved_models/{CFG["exp_name"]}', exist_ok=True)
device = torch.device(CFG['cuda'] if torch.cuda.is_available() else 'cpu')

utils.seed_everything(CFG['SEED']) ## Seed 고정
####################################################################################

# Label Converter
####################################################################################
if 'CTC' in CFG['Prediction']:
    converter = utils.CTCLabelConverter(CFG['character'])
else:
    converter = utils.AttnLabelConverter(CFG['character'])
CFG['num_class'] = len(converter.character) ## the number of class(syllable)
####################################################################################

# 모델 생성
####################################################################################
model = clova_model.Model(CFG)
####################################################################################

# 가중치 초기화
####################################################################################
for name, param in model.named_parameters():
    if 'localization_fc2' in name:
        print(f'Skip {name} as it is already initialized')
        continue
    try:
        if 'bias' in name:
            init.constant_(param, 0.0)
        elif 'weight' in name:
            init.kaiming_normal_(param)
    except Exception as e:  # for batchnorm.
        if 'weight' in name:
            param.data.fill_(1)
        continue
####################################################################################

# pretrained model 로드
####################################################################################
if CFG['saved_model']:
    print(f'loading pretrained model from {CFG["saved_model"]}')
    if CFG['FT']:
        model.load_state_dict(torch.load(CFG['saved_model']), strict=False)
    else:
        model.load_state_dict(torch.load(CFG['saved_model']))
####################################################################################

# config 저장
####################################################################################
with open(f'./saved_models/{CFG["exp_name"]}/config_{now}.txt', 'w', encoding='UTF-8') as f:
    for name, info in CFG.items():
        f.write(f'{name} : {info}\n')
####################################################################################

# loss 생성
####################################################################################
if 'CTC' in CFG['Prediction']:
    criterion = nn.CTCLoss(zero_infinity=True).to(device)
else:
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
####################################################################################

# optimizer 생성
####################################################################################
filtered_parameters = []
for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_parameters.append(p)

if CFG['adam']:
    optimizer = torch.optim.Adam(filtered_parameters, lr=CFG['lr'], betas=(CFG['beta1'], 0.999))
else:
    optimizer = torch.optim.Adadelta(filtered_parameters, lr=CFG['lr'], rho=CFG['rho'], eps=CFG['eps'])
####################################################################################

# dataloader 생성
####################################################################################
train_dataset = CustomDataset(train['img_path'].values, train['label'].values, CFG, train_mode=True)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])

val_dataset = CustomDataset(val['img_path'].values, val['label'].values, CFG, train_mode=False)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])
####################################################################################

# training
####################################################################################
infer_model = custom_train.train(model, optimizer, criterion, train_loader, val_loader, converter, CFG, device)
####################################################################################

# test
####################################################################################
test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['img_path'].values, None, CFG, train_mode=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

predictions = custom_test.inference(infer_model, test_loader, converter, CFG, device)
# predictions = custom_test.inference(model, test_loader, converter, CFG, device)

submit = pd.read_csv('./sample_submission.csv')
submit['label'] = predictions

submit.to_csv(f'./saved_models/{CFG["exp_name"]}/submission_{now}.csv', index=False, encoding="utf-8-sig")
####################################################################################