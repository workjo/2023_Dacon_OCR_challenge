import datetime as dt


CFG = {
    'IMG_HEIGHT_SIZE':32,                       ## the height of the input image
    'IMG_WIDTH_SIZE':100,                       ## the width of the input image
    'EPOCHS':500,                               ## epoch
    'lr':1,                                     ## learning rate, default=1.0 for Adadelta
    'BATCH_SIZE':512,                           ## batch size
    'cuda':'cuda',                              ## 'cuda' if you want to use gpu, else 'cpu'
    'NUM_WORKERS':1,                            ## 본인의 GPU, CPU 환경에 맞게 설정
    'SEED':41,                                  ## random seed
    'Transformation':'TPS',                     ## Transformation stage. None|TPS
    'FeatureExtraction':'ResNet',               ## FeatureExtraction stage. VGG|RCNN|ResNet
    'SequenceModeling':'BiLSTM',                ## SequenceModeling stage. None|BiLSTM
    'Prediction':'Attn',                        ## Prediction stage. CTC|Attn
    'num_fiducial':20,                          ## number of fiducial points of TPS-STN
    'input_channel':3,                          ## the number of input channel of Feature extractor
    'output_channel':512,                       ## the number of output channel of Feature extractor
    'hidden_size':256,                          ## the size of the LSTM hidden state
    'batch_max_length':11,                      ## maximum-label-length
    'rho':0.95,                                 ## decay rate rho for Adadelta. default=0.95
    'eps':1e-8,                                 ## eps for Adadelta. default=1e-8
    'adam':False,                               ## Whether to use adam (default is Adadelta)
    'beta1':0.9,                                ## beta1 for adam. default=0.9
    'grad_clip':5,                              ## gradient clipping value. default=5
    'saved_model':'',                           ## location of the pretrained model. default=''
    'FT':False,                                 ## finetuning. default=False
    'aug':False                                 ## gaussian blur for image augmentation (ratio:0.5)
}

now = dt.datetime.now().strftime("%y%m%d_%H%M")

CFG['exp_name'] = f'{CFG["Transformation"]}-{CFG["FeatureExtraction"]}-{CFG["SequenceModeling"]}-{CFG["Prediction"]}'
CFG['exp_name'] += f'-Seed{CFG["SEED"]}_{now}'