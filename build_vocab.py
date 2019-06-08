import pickle
import argparse
from collections import Counter
from data_loader import CocoDataset
import gensim
import torch


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(args):
    coco = CocoDataset(annotation_path=args.caption_path)
    counter = Counter()

    for i, id in enumerate(coco.ids):
        tokens = coco.captions[id]['caption']
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, coco.__len__()))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= args.threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>') # padding을 0으로 지정
    vocab.add_word('<start>') # start of sentence
    vocab.add_word('<end>') # end of sentence
    vocab.add_word('<unk>') # unknown

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab


# class Word2VecVocab(object):
#     """Simple vocabulary wrapper."""
#     def __init__(self, model, args):
#         self.word2vec = {}
#         self.model = model
#         self.embedding_size = args.embedding_size
#
#     def add_word(self, word):
#         if not word in self.word2vec:
#             self.word2vec[word] = self.model[word]
#
#     def __call__(self, word):
#         if not word in self.word2vec:
#             return torch.zeros((self.embedding_size,))
#         return self.word2vec[word]
#
#     def __len__(self):
#         return len(self.word2vec)
#
#
# def build_word2vec(args):
#
#     """if you want to train word2vec model, use annotated code below"""
#
#     coco = CocoDataset(annotation_path=args.caption_path)
#     token_all = []
#
#     #Todo: Word2Vec 다 쪼개져서 학습됨 -> 주피터에서 확인
#     #Todo: Word2Vec에 없는 단어는 어떻게 처리? 그냥 FastText로 할지...
#     for i, id in enumerate(coco.ids):
#         tokens = coco.captions[id]['caption']
#         sentence_len = len(tokens)
#         token_all.extend(['<pad>' for i in range(coco.caption_len - sentence_len)])
#         token_all.append('<start>')
#         token_all.extend(tokens)
#         token_all.append('<end>')
#
#     model = gensim.models.Word2Vec(token_all, size=args.embedding_size, window=3, min_count=2)
#     print('trained word2vec model completely.')
#
#     model.wv.save_word2vec_format(args.model_path, binary=True)
#     model = gensim.models.KeyedVectors.load_word2vec_format(args.model_path, binary=True)
#
#     vocab = Word2VecVocab(model, args)
#     for word in list(model.vocab.keys()):
#         vocab.add_word(word)
#
#     return vocab


def main(args):
    vocab = build_vocab(args)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(args.vocab_path))

    # w2v_vocab = build_word2vec(args)
    # with open(args.word2vec_path, 'wb') as f:
    #     pickle.dump(vocab, f)
    # print("Total vocabulary size: {}".format(len(vocab)))
    # print("Saved the vocabulary wrapper to '{}'".format(args.word2vec_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--embedding_size', type=int, default=10,
                        help='embedding size for Word2Vec')
    parser.add_argument('--model_path', type=str, default='./data/word2vec_model.bin',
                        help='path for pretrained Word2Vec model')
    parser.add_argument('--word2vec_path', type=str, default='./data/word2vec_vocab.pkl',
                        help='path for saving Word2Vec')
    args = parser.parse_args()
    main(args)
