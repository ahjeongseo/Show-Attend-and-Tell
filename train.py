import argparse
import os
import pickle
from data_loader import get_loader
from model import EncoderCNN, DecoderLSTM
from build_vocab import Vocabulary
from logger import Logger
from utils import *
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def to_var(x):
    if torch.cuda.is_available():
        x.cuda()
    return Variable(x)

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # image preprocessing and data loading
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
        ])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = {}
    data_loader['train'] = get_loader(args.train_annote_path, args.train_image_path, vocab, transform,
                               args.batch_size,shuffle=True, num_workers=args.num_workers)
    data_loader['val'] = get_loader(args.val_annote_path, args.val_image_path, vocab, transform,
                             args.batch_size,shuffle=True, num_workers=args.num_workers)

    total_train_step = len(data_loader['train'])
    total_val_step = len(data_loader['val'])

    # build models and load checkpoint
    encoder = EncoderCNN()
    decoder = DecoderLSTM(len(vocab), args.embed_size, args.feature_num, args.hidden_size, args.attention_dim, args.num_layers)

    start_epoch = 0
    if args.pretrained_checkpoint > 0:
        filename = 'checkpoint_' + str(args.pretrained_checkpoint) + '.pth.tar'
        checkpoint = torch.load(os.path.join(args.model_path, filename))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu4']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        print("Cuda is enabled..")

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=args.learning_rate, weight_decay=args.weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # write log in tensorboard
    logger = Logger('./logs')

    # training
    for epoch in range(start_epoch, args.num_epochs):
        # train
        avg_loss = 0.0
        encoder.train()
        decoder.train()
        for i, (images, captions, lengths) in enumerate(data_loader['train']):
            images = to_var(images)
            captions = to_var(captions)

            # Todo: accuracy 구해야 함???
            targets = captions[:, 1:] # erase <start>
            targets, _ = pack_padded_sequence(targets, lengths, batch_first=True)

            features = encoder(images)
            outputs, alphas = decoder(features, captions, lengths)
            outputs, _ = pack_padded_sequence(outputs, lengths, batch_first=True)

            loss = criterion(outputs, targets)
            loss += args.loss_param_lambda * ((1 - alphas.sum(dim=1)) ** 2).mean()
            # 논문에서, alpha들의 합이 1이 되도록 강제하지 않음으로써 attention효과 증가(4.2.1절)

            # 역전파 실행 전 변화도를 0으로 초기화
            decoder.zero_grad()
            encoder.zero_grad()

            avg_loss += loss.item()
            # backprop 계산
            loss.backward()
            # 매개변수 값 갱신
            encoder_optimizer.step()
            decoder_optimizer.step()

            if i % args.log_step == 0:
                print('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Perplexity: %.4f'
                      % (epoch+1, args.num_epochs, i, total_train_step, loss.item(), np.exp(loss.item())))

        avg_loss /= (args.batch_size * total_train_step)
        print('Epoch [%d/%d], Average Train Loss: %.4f, Average Train Perplexity: %.4f'
              % (epoch+1, args.num_epochs, avg_loss, np.exp(avg_loss)))

        #val
        avg_loss = 0.0
        encoder.eval()
        decoder.eval()
        for i, (images, captions, lengths) in enumerate(data_loader['val']):
            # Todo: pytorch tutorial -> variable/autograd 의미
            images = to_var(images)
            captions = to_var(captions)

            features = encoder(images)
            outputs, alphas = decoder(features, captions, lengths)

            loss = criterion(outputs, targets)
            loss += args.loss_param_lambda * ((1 - alphas.sum(dim=1)) ** 2).mean()

            avg_loss += loss.item()

            if i % args.log_step == 0:
                print('Epoch [%d/%d], Val Step [%d/%d], Loss: %.4f, Perplexity: %.4f'
                      % (epoch+1, args.num_epochs, i, total_val_step, loss.item(), np.exp(loss.item())))

        avg_loss /= (args.batch_size * total_val_step)
        print('Epoch [%d/%d], Average Val Loss: %.4f, Average Val Perplexity: %.4f'
              % (epoch+1, args.num_epochs, avg_loss, np.exp(avg_loss)))

        # before blue score, clean sentences -> remove <start>, <pad>
        # overfit warning

        # improvement check
        epochs_since_improvement
        bleu4
        is_best
        # save checkpoint
        save_checkpoint(epoch+1, epochs_since_improvement, bleu4, encoder.state_dict(), decoder.state_dict(), args.model_path, is_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./data/models')
    parser.add_argument('--crop_size', type=int, default=224) # resize:256
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl')
    parser.add_argument('--train_annote_path', type=str, default='./data/annotations/train/captions.json')
    parser.add_argument('--val_annote_path', type=str, default='./data/annotations/val/')
    parser.add_argument('--train_image_path', type=str, default='./data/images/train')
    parser.add_argument('--val_image_path', type=str, default='./data/images/val')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pretrained_checkpoint', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=20)

    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--loss_param_lambda', type=float, default=1.)
    args = parser.parse_args()
    print(args)
    main(args)
