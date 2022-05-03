import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.utils.data as data_utils
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models import EncoderCNN, DecoderRNN, CNNtoRNN, Attention
from torchvision import datasets, transforms
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import nltk
from sklearn.model_selection import train_test_split
from PIL import Image
from utils import build_vocab, train_val_test_split, trainModel, testModel, load_saved_model
from models import encoder, attention, decoder, decoder1, fullModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

base_path = '/content/drive/MyDrive/DL_Project/flickr8k.zip (Unzipped Files)'
base_path = '/home/bapi/ntk/testing'
      

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
image_transforms = transforms.Compose([
    transforms.Resize(226),                     
    transforms.RandomCrop(224),
    transforms.ToTensor(),                               
    # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

dataset =  Flickr8kDataset(base_path=base_path, tokenizer=tokenizer, image_transform = image_transforms)
captions = dataset.get_captions()

vocab, vocab_size, inverse_vocab, key_to_ind = build_vocab(captions, tokenizer)
list_vocab = torch.tensor(list(inverse_vocab.keys()))
list_vocab = list(map(int,list_vocab))

train_data, val_data, test_data = train_val_test_split(dataset, 0.7,0.15,0.15)
batch_size = 256
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data , batch_size=batch_size, shuffle=True)

embed_size = 512
resnet = models.resnet18(pretrained=True, progress=True)
resnet = resnet.to(device = device)
num_layers = 2
max_len = 32
unfreeze1, unfreeze2 = 10, 0
num_epochs = 6
learning_rate = 1e-3

model = fullModel(embed_size, resnet, vocab_size, num_layers, vocab, tokenizer, max_len, unfreeze1, unfreeze2, inverse_vocab, vocab_list=list_vocab)
model = model.to(device=device)
criteria = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
base_path +'/'
model_name = 'resnet_distilbert_mode_'
model_name += 'unfreeze1_' + str(unfreeze1) + '_unfreeze2_' +str(unfreeze2) + '_gru_layers_' + str(num_layers)
print("Epochs : " + str(num_epochs))
print("LR : " + str(learning_rate))
print("Batch size : " + str(batch_size))
print("Model name : " + model_name)

trainModel(model, optimizer, device, criteria, train_loader, val_loader, test_loader, num_epochs, loss_plot_path = base_path, model_name = model_name, model_save_path = base_path)
testModel(model, test_loader, criteria, is_val = False)
