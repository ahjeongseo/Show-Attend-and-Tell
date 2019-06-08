import torch
from torch import nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # input size : 224*224*3
        vgg = models.vgg19_bn(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.children())[0][:-1]) # size: batch*512*14*14
        # 이미 구현상 마지막 단 batch norm, relu가 되어있으므로 layer 더 추가할 필요 없음

        # 막단의 conv2d, batchnorm, relu 세 가지만 fine tuning
        fine_tune_start = len(self.vgg) - 3
        for i, child in enumerate(self.vgg):
            for param in child.parameters():
                param.requires_grad = False if i < fine_tune_start else True

    def forward(self, images):
        features = self.vgg(images)
        features = torch.flatten(features, start_dim=-2, end_dim=-1) # size: batch*512*196
        return features


class Attention(nn.Module):
    def __init__(self, feature_num, hidden_size, attention_dim, num_layers):
        super(Attention, self).__init__()
        self.linear_1 = nn.Linear(feature_num + hidden_size*num_layers, attention_dim)
        self.linear_2 = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def soft_attention(self, encoder_out, h_prev):
        all_output = []
        features = encoder_out.view(-1, encoder_out.shape[0], encoder_out.shape[1]) # size: 196*batch*512
        h_prev = h_prev.view(h_prev.shape[1], -1, h_prev.shape[2]) # size: batch*num_layers*512(hidden_size)
        for feature in features:
            att_input = torch.cat([feature.unsqueeze(1), h_prev], dim=1) # size: batch*(1+num_layers)*512
            att_flat = torch.flatten(att_input, start_dim=1, end_dim=2)
            att_output = self.linear_2(self.tanh(self.linear_1(att_flat))) # size: batch*512
            all_output.append(att_output)
        alpha = self.softmax(torch.cat(all_output, dim=1)) # size: batch*196
        z = (encoder_out * alpha.unsqueeze(1)).sum(dim=2) # size: batch*512
        return z, alpha

    #Todo: Hard Attention

    def forward(self, encoder_out, h_prev):
        z, alpha = self.soft_attention(encoder_out, h_prev)
        return z, alpha


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, feature_num, hidden_size, attention_dim, num_layers):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(feature_num, hidden_size, attention_dim, num_layers)
        self.f_beta = nn.Linear(hidden_size*num_layers, feature_num)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(embed_size + feature_num, hidden_size, num_layers, batch_first=True)
        # input/output: (batch, seq, feature)
        self.dropout = nn.Dropout(p=0.5)
        # 논문에서 regularizer로 dropout과 BLEU score early stopping 사용
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        return h0, c0

    def forward(self, features, captions, lengths):
        batch_size = features.shape[0]
        h, c = self.init_hidden(batch_size)

        embedding = self.embed(captions) # size: caption_len*embed_size
        decode_lengths = [len - 1 for len in lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), self.attention_dim)
        if torch.cuda.is_available():
            predictions = predictions.cuda()
            alphas = alphas.cuda()

        for i in range(max(decode_lengths)):
            batch_idx_to_decode = sum([l > i for l in decode_lengths])
            z, alpha = self.attention(features[:batch_idx_to_decode], h[:batch_idx_to_decode])
            h_gate = h[:batch_idx_to_decode, :, :]
            h_gate = torch.flatten(h_gate.view(h_gate.shape[1], h_gate.shape[0], -1), start_dim=1, end_dim=2)
            z = z * self.sigmoid(self.f_beta(h_gate))
            # 논문에서, attention output z에 gating scalar beta를 곱함으로써 성능 상승(4.2.1절)
            lstm_input = torch.cat([embedding[:batch_idx_to_decode, i, :], z[:batch_idx_to_decode, :]], dim=1)

            #Todo: lstm input 3차원으로 변형하여 대입 -> num_layers 수에 맞게 매칭??
            h, c = self.lstm(lstm_input, (h[:batch_idx_to_decode, :, :], c[:batch_idx_to_decode, :, :]))
            output = self.linear(self.dropout(h))
            predictions[:batch_idx_to_decode, i, :] = output
            alphas[:batch_idx_to_decode, i, :] = alpha

        return predictions, alphas

    #Todo: Decoder forward부분 -> 이후 training과 연결하며 개선할 여지가 있는지??