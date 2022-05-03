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
spacy_eng = spacy.load("en")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flickr8kDataset(Dataset):
    def __init__(self, base_path, tokenizer, image_transform=None, text_transform=None):
        self.base_path = base_path
        self.image_path = os.path.join(self.base_path, 'Images')
        self.captions = pd.read_csv(os.path.join(self.base_path, 'captions.txt'))
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.tokenizer = tokenizer
        
        self.images = self.captions["image"]
        self.captions = self.captions["caption"]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        img = Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB")
        
        if self.image_transform is not None:
            img = self.image_transform(img)
        # print(caption)
        inputs = self.tokenizer.encode_plus(caption,
            add_special_tokens=True,
            padding='max_length',
            max_length = 32,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        req_dic = {
            "image" : img,
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long)
        }

        return req_dic
    
    def visualize(self):

      for i in range(5):
        img = Image.open(os.path.join(self.image_path, self.images[i])).convert("RGB")
        caption = self.captions[i]
        print(caption)
        plt.imshow(img)

    def get_captions(self):
      return list(self.captions)


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ] 


class CapsCollate:
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets