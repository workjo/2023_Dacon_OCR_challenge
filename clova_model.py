"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Original Code
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/model.py
modified by Lee Jaeoh
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, CFG):
        super(Model, self).__init__()
        self.CFG = CFG
        self.stages = {'Trans': CFG['Transformation'], 'Feat': CFG['FeatureExtraction'],
                       'Seq': CFG['SequenceModeling'], 'Pred': CFG['Prediction']}

        """ Transformation """
        if CFG['Transformation'] == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=CFG['num_fiducial'], I_size=(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']), I_r_size=(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']), I_channel_num=CFG['input_channel'])
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if CFG['FeatureExtraction'] == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(CFG['input_channel'], CFG['output_channel'])
        elif CFG['FeatureExtraction'] == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(CFG['input_channel'], CFG['output_channel'])
        elif CFG['FeatureExtraction'] == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(CFG['input_channel'], CFG['output_channel'])
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = CFG['output_channel']  # int(IMG_HEIGHT_SIZE/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (IMG_HEIGHT_SIZE/16-1) -> 1

        """Sequence modeling"""
        if CFG['SequenceModeling'] == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, CFG['hidden_size'], CFG['hidden_size']),
                BidirectionalLSTM(CFG['hidden_size'], CFG['hidden_size'], CFG['hidden_size']))
            self.SequenceModeling_output = CFG['hidden_size']
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if CFG['Prediction'] == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, CFG['num_class'])
        elif CFG['Prediction'] == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, CFG['hidden_size'], CFG['num_class'])
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.CFG['batch_max_length'])

        return prediction
