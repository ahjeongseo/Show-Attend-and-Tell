import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
import nltk
import os


class CocoDataset(data.Dataset):
    def __init__(self, annotation_path, image_path, vocab, transform=None):
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.load_dataset()
        self.images, self.captions, self.ids = self.load_dataset()
        self.vocab = vocab
        if transform == None:
            transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
        self.transform = transform
        global pad_idx
        pad_idx = self.vocab('<pad>')

    def load_dataset(self):
        json_file = json.load(open(self.annotation_path, 'r'))
        images = {}
        captions = {}
        ids = []

        for image in json_file['images']:
            images[image['id']] = image

        for caption in json_file['annotations']:
            captions[caption['id']] = caption
            ids.append(caption['id'])

        return images, captions, ids

    def __getitem__(self, index):
        caption_id = self.ids[index]
        image_id = self.captions[caption_id]['image_id']
        image_filename = self.images[image_id]['file_name']

        image = Image.open(os.path.join(self.image_path, image_filename)).convert('RGB')
        image = self.transform(image)

        sentence = self.captions[caption_id]['caption']
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())

        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True) # sort는 내부에서 정렬, sorted는 정렬된 값을 반환
    images, captions = zip(*data)
    max_len = len(captions[0])

    images = torch.stack(images, 0)
    targets = []
    lengths = []
    for caption in captions:
        lengths.append(len(caption))
        padding = torch.FloatTensor([pad_idx for i in range(max_len - len(caption))])
        targets.append(torch.cat([padding, caption], dim=0))
    targets = torch.stack(targets).long()

    return images, targets, lengths


def get_loader(annotation_path, image_path, vocab, transform, batch_size, shuffle, num_workers):
    coco = CocoDataset(annotation_path, image_path, vocab, transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
