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

def show_image(inp, title=None, idx = 0, path = ''):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    
    if(idx!=0):
        plt.savefig(path+"prediction"+str(idx)+'.png')
    # plt.pause(0.1)
    plt.show()

def build_vocab(captions, tokenizer):
      vocab = {}
      inverse_vocab = {}
      vocab_size = 0
      key_to_ind = {}

      for caption in captions:
        inputs = tokenizer.encode_plus(caption,
          add_special_tokens=True,
          padding='max_length',
          max_length = 32,
          truncation=True,
          return_attention_mask=True,
          return_tensors="pt"
        )
        req_dic = {
              "ids": torch.tensor(inputs["input_ids"][0], dtype=torch.long),
              "mask": torch.tensor(inputs["attention_mask"][0], dtype=torch.long)
        }
        words = tokenizer.convert_ids_to_tokens(req_dic["ids"])
        vals = req_dic["ids"]
        
        for i in range(len(words)):
          if(words[i] not in vocab):
              vocab[words[i]] = int(vals[i])
              inverse_vocab[int(vals[i])] = words[i]
              key_to_ind[int(vals[i])] = vocab_size
              vocab_size += 1

      return vocab, vocab_size, inverse_vocab, key_to_ind

def train_val_test_split(data, train,val,test):
    """ a function that will get dataset and training dataset fraction as input and return x_train, x_test, y_train, y_test """
    
    train_samples=len(data)*train//(train+test+val)
    val_samples=len(data)*val//(train+test+val)
    train_samples = int(train_samples)
    val_samples = int(val_samples)
    test_samples = len(data) - train_samples - val_samples

    train_data, val_data,test_data = torch.utils.data.random_split(data, [train_samples, val_samples, test_samples])
    
    return train_data,val_data,test_data

def trainModel(model, optimizer, device, criteria, train_loader, val_loader, test_loader, epochs, loss_plot_path = None, model_name = None, model_save_path = None):
    model.train()
    batch_size = 0
    loss_vs_epochs = []
    val_loss_vs_epochs = []

    for epoch in range(epochs):

        print("Epoch "+str(epoch+1))
        total_samples = 0
        cur_epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            print(batch_idx)
            ids = batch["ids"]
            mask = batch["mask"]
            ids = torch.squeeze(ids)
            mask = torch.squeeze(mask)
            total_samples += len(ids)
            image = batch["image"].to(device=device)
            ids = ids.to(device=device, dtype=torch.long)
            mask = mask.to(device=device, dtype=torch.long)
            
            words_probability, captions_strings = model(image, ids, mask)

            for i in range(len(ids)):
              for j in range(len(ids[i])):
                  ids[i][j] = torch.tensor(key_to_ind[int(ids[i][j])])

            loss = criteria(words_probability, ids)
            print("Batch loss : " + str(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            cur_epoch_loss += loss.item()
            
            optimizer.step()
            
        loss_vs_epochs.append(cur_epoch_loss/total_samples)
        
        print("Train loss : " + str(loss_vs_epochs[-1]))

        print()

    if(loss_plot_path):

        plt.plot(loss_vs_epochs, c='r')
        plt.plot(val_loss_vs_epochs, c='b')
        plt.legend(['Train loss', 'Validation loss'])
        if(model_name):
            plt.title("Model : "+model_name)
        
        plt.savefig(loss_plot_path + model_name + '_training_loss.png')
        plt.show()
        plt.clf()

    if(model_save_path):
        torch.save(model.state_dict(), model_save_path + model_name + '.pkl')

def testModel(model, test_loader, criteria, is_val = False):

    val_loss_vs_epochs = []
    total_samples = 0
    cur_epoch_loss = 0

    model.eval()

    for batch_idx, batch in enumerate(tqdm(test_loader)):
        
        ids = batch["ids"]
        mask = batch["mask"]
        ids = torch.squeeze(ids)
        mask = torch.squeeze(mask)
        total_samples += len(ids)
        image = batch["image"].to(device=device)
        ids = ids.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.long)
        
        words_probability, captions_strings = model(image, ids, mask, test=True)
        
        print(captions_strings)
        for i in range(len(ids)):
          for j in range(len(ids[i])):
              ids[i][j] = torch.tensor(key_to_ind[int(ids[i][j])])

        loss = criteria(words_probability, ids)
        cur_epoch_loss += loss.item()

    model.train()

    if(is_val):
        loss = cur_epoch_loss/total_samples
        print("Validation loss : "+str(loss))
        return loss

    else:
        print()

        loss = cur_epoch_loss/total_samples
        print("Testing loss: "+str(loss))
        return loss

def load_saved_model(model_name : str, device : str, model: torch.nn, base_path : str):
    model.load_state_dict(torch.load(base_path + model_name, map_location=torch.device(device)))
    model.eval()
    return model


def get_caps_from(features_tensors):

    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps,alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0],title=caption)
    
    return caps,alphas


def save_model(model,num_epochs):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size':len(dataset.vocab),
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict()
    }

    torch.save(model_state,'attention_model_state.pth')
