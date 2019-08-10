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
    def __init__(self, feature_num, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(feature_num, attention_dim)
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def soft_attention(self, features, h_prev):
        features = features.view(features.shape[0], -1, features.shape[1]) # size: batch*512*196 -> batch*196*512
        en_att = self.encoder_att(features) # size: batch*196*512
        de_att = self.decoder_att(h_prev) # size: batch*512
        full_att = self.full_att(self.tanh(en_att + de_att.unsqueeze(1))).squeeze(2) # size: batch*196
        alpha = self.softmax(full_att) # size: batch*196
        z = (features * alpha.unsqueeze(2)).sum(dim=1) # size: batch*512
        return z, alpha

    #Todo: Hard Attention

    def forward(self, encoder_out, h_prev):
        z, alpha = self.soft_attention(encoder_out, h_prev)
        return z, alpha


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, feature_num, hidden_size, attention_dim):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size #512
        self.attention_dim = attention_dim #512 : image 지역 개수(filter 수)

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(feature_num, hidden_size, attention_dim)
        self.f_beta = nn.Linear(hidden_size, feature_num)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTMCell(embed_size + feature_num, hidden_size, bias=True)
        # LSTM 매 step의 cell마다 attention 값 concat하여 계산 -> nn.LSTM대신 nn.LSTMCell사용하여 하나씩 계산
        # Todo: LSTM hidden layer >= 2 인 경우
        self.dropout = nn.Dropout(p=0.5)
        # 논문에서 regularizer로 dropout과 BLEU score early stopping 사용
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.init_h = nn.Linear(feature_num, hidden_size)
        self.init_c = nn.Linear(feature_num, hidden_size)

    def init_hidden(self, features):
        # 논문에서 첫 hidden vector : annotation vector들의 평균
        mean_features = features.mean(dim=2) #size: batch*512
        h0 = self.init_h(mean_features) # size: batch*512
        c0 = self.init_c(mean_features) # size: batch*512

        return h0, c0

    def forward(self, features, captions, lengths):
        batch_size = features.shape[0]
        feature_dim = features.shape[2] # 196 : 하나의 filter의 dimension
        h, c = self.init_hidden(features)

        embedding = self.embed(captions) # size: batch*caption_len*512
        decode_lengths = [len - 1 for len in lengths]
        # decoding시 마지막의 <end> token 빼고 decode

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), feature_dim)
        if torch.cuda.is_available():
            predictions = predictions.cuda()
            alphas = alphas.cuda()

        z, alpha = self.attention(features, h)
        z = z * self.sigmoid(self.f_beta(h))
        lstm_input = torch.cat([embedding, z.squeeze(1)], dim=2)
        h, c = self.lstm(lstm_input, (h, c))

        for i in range(max(decode_lengths)):
            batch_idx = sum([l > i for l in decode_lengths])
            z, alpha = self.attention(features[:batch_idx], h[:batch_idx])
            z = z * self.sigmoid(self.f_beta(h[:batch_idx]))
            # 논문에서, attention output z에 gating scalar beta를 곱함으로써 성능 상승(4.2.1절)

            lstm_input = torch.cat([embedding[:batch_idx, i, :], z[:batch_idx]], dim=1) # size: batch*512
            h, c = self.lstm(lstm_input, (h[:batch_idx], c[:batch_idx]))
            output = self.linear(self.dropout(h)) # size: batch*vocab_size
            predictions[:batch_idx, i, :] = output
            alphas[:batch_idx, i, :] = alpha # size: batch*(sentence_len)*196

        return predictions, alphas
