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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import nltk
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from PIL import Image
from dataset_class import FlickrDataset, Vocabulary, CapsCollate
from utils import show_image, get_caps_from, save_model, train_val_test_split
from models import EncoderDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

base_path = '/content/drive/MyDrive/DL_Project/flickr8k.zip (Unzipped Files)'
base_path = '/home/bapi/ntk/testing'

spacy_eng = spacy.load("en")

v = Vocabulary(freq_threshold=1)
data_location = base_path



data_location = base_path
transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    # T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])


dataset =  FlickrDataset(
    root_dir = data_location+"/Images",
    captions_file = data_location+"/captions.txt",
    transform=transforms
)
pad_idx = dataset.vocab.stoi["<PAD>"]


batch_size = 16

train_data, val_data, test_data = train_val_test_split(dataset, 0.7,0.15,0.15)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True))
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True))
val_loader = DataLoader(dataset=val_data , batch_size=batch_size, shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True))

vocab_size = len(dataset.vocab)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#Hyperparams
embed_size=300
vocab_size = len(dataset.vocab)
attention_dim=256
encoder_dim=2048
decoder_dim=512
learning_rate = 3e-4

model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

req_dict = torch.load('/home/bapi/ntk/testing/attention_model_state.pth')
model.load_state_dict(req_dict['state_dict'])
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.eval()

base_path += '/prediction/'

for batch_idx, batch in enumerate(tqdm(test_loader)):
    img, _ = batch
    features = model.encoder(img[0:1].to(device))
    caps,alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
    caption = ' '.join(caps)
    show_image(img[0],title=caption, idx=batch_idx, path=base_path)

