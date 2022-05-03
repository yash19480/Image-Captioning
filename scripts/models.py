import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class encoder(nn.Module):
    def __init__(self, embed_size, model, unfreeze=10):
        super(encoder, self).__init__()
        
        self.resnet = model
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.total_trainable_layers = 0
        self.freeze_count = 0

        for name, param in self.resnet.named_parameters():
          if param.requires_grad:
              self.total_trainable_layers += 1
              if(self.freeze_count + unfreeze < 60):
                  param.requires_grad = False
                  self.freeze_count += 1
        
        print("Total trainable distil bert layers are : " + str(self.total_trainable_layers))
        print("Layers freezed = "+str(self.freeze_count))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1)) #(B, 7*7, 512)
        return self.dropout(self.relu(features))

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha, attention_weights

class decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, vocab, tokenizer):
        super(decoder, self).__init__()
        self.tokenizer = tokenizer
        self.encoder_dim = 512
        self.embed_dim = 512
        self.attention_dim = 256
        self.decoder_dim = 256
        self.vocab = vocab
        self.vocab_size = vocab_size

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)

        self.decode_step = nn.GRU(input_size=self.embed_dim + self.encoder_dim, hidden_size=self.decoder_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

    # def forward(self, features, captions):
    def forward(self, features):
        enc_attn = self.attention(features)
        print(enc_attn.size())

        return enc_attn

class decoder1(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers, vocab,tokenizer, max_len = 32, unfreeze = 10):
        super(decoder1, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_len = max_len
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True)
        self.distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased",).to(device=device)
        self.tokenizer = tokenizer
        self.total_trainable_layers_distil_bert = []
        self.freeze_count = 0

        for name, param in self.distil_bert.named_parameters():
          if param.requires_grad:
              self.total_trainable_layers_distil_bert.append(name)

              if(self.freeze_count + unfreeze < 100):
                  param.requires_grad = False
                  self.freeze_count += 1

        print("Total trainable distil bert layers are : " + str(len(self.total_trainable_layers_distil_bert)))
        print("Layers freezed = "+str(self.freeze_count))

        self.bert_to_enc_dim = nn.Linear(768, 512)
        self.l1 = nn.Linear(2*512, self.vocab_size)

    def forward(self, enc_embed, ids, mask):

        embed = self.distil_bert(ids, attention_mask = mask)

        embed = embed[0]
        embed = self.bert_to_enc_dim(embed)

        all_words = []

        for i in range(self.max_len):
            cur_tens = torch.tensor([i]).to(device=device)
            word_embed = torch.index_select(embed,dim=1, index=cur_tens)

            enc_cat = torch.cat([word_embed, enc_embed], dim=1)

            e_cat, _ = self.gru(enc_cat)

            e_cat = e_cat.max(dim = 1)[0]
            e_cat = torch.squeeze(e_cat)

            e_cat = F.leaky_relu(self.l1(e_cat))
            e_cat = F.softmax(e_cat, dim=1)
            # max_prob = torch.argmax(e_cat, dim=1)
            
            all_words.append(e_cat)

        final_words = torch.stack(all_words, dim=2)   ## index of words
        return final_words
        

    def evaluate_test(self, enc_embed, ids, mask, i):

        if(i==0):
            embed = self.distil_bert(ids, attention_mask = mask)
            embed = embed[0]
            embed = self.bert_to_enc_dim(embed)
            cur_tens = torch.tensor([i]).to(device=device)
            word_embed = torch.index_select(embed,dim=1, index=cur_tens)

        else:
            
            embed = self.distil_bert(ids, attention_mask = mask)
            embed = embed[0]
            word_embed = self.bert_to_enc_dim(embed)

        enc_cat = torch.cat([word_embed, enc_embed], dim=1)
        e_cat, _ = self.gru(enc_cat)

        e_cat = e_cat.max(dim = 1)[0]
        e_cat = torch.squeeze(e_cat)

        e_cat = F.leaky_relu(self.l1(e_cat))
        e_cat = F.softmax(e_cat, dim=1)
        
        return e_cat    ## single word softmax

class fullModel(nn.Module):
    def __init__(self, embed_size, resnet, vocab_size, num_layers, vocab, tokenizer, max_len, unfreeze1, unfreeze2, inverse_vocab, vocab_list):
        super(fullModel, self).__init__()
        self.inverse_vocab = inverse_vocab
        self.vocab_size = vocab_size
        self.vocab_list = vocab_list
        self.embed_size = embed_size
        self.max_len = max_len
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.enc = encoder(embed_size, resnet, unfreeze=unfreeze1)
        self.dec = decoder1(embed_size, vocab_size, num_layers, vocab, tokenizer, max_len = max_len, unfreeze = unfreeze2)

    def forward(self, image, ids, mask, test = False):

        if(not test):
            enc_embed = self.enc(image)
            words_probability = self.dec(enc_embed, ids, mask)
            words_pred = torch.argmax(words_probability, dim=1)
            captions_strings = self.index_to_captions(words_pred, mask)

            return words_probability, captions_strings
        
        else:
            enc_embed = self.enc(image)
            distinct_words = []
            total_probability = []
            prev_word = None

            for i in range(self.max_len):

              if(prev_word is not None):
                mask = torch.ones(ids.size(0),1)
                for i in range(len(ids)):
                  ids[i] = torch.tensor(self.vocab_list[int(prev_word[i])])
            
              words_probability = self.dec.evaluate_test(enc_embed, ids, mask, i)
              words_pred = torch.argmax(words_probability, dim=1)
              prev_wrod = words_pred

              distinct_words.append(words_pred)
              total_probability.append(words_probability)

            words_probability = torch.stack(total_probability, dim=2)
            all_words = torch.stack(distinct_words, dim=1)

            captions_strings = self.index_to_captions(all_words, mask)
            return words_probability, captions_strings

    def index_to_captions(self, words_tensor, mask):
        #(batch, words)
        captions = []

        for i in range(len(words_tensor)):
            cur_words = []
            ind = mask[i].argmin()
            for j in range(ind):
                cur_words.append(self.inverse_vocab[int(self.vocab_list[int(words_tensor[i][j])])])
            
            captions.append(' '.join(cur_words))
        
        return captions

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features

#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha,attention_weights
        
#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
        
    
    def forward(self, features, captions):
        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
                
        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas
    
    def generate_caption(self,features,max_len=20,vocab=None):
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)

        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
        
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        return [vocab.itos[idx] for idx in captions],alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

