import nltk
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import json
from vocabulary import Vocabulary


def get_loader(transform, mode='train', batch_size=1, vocab_threshold=None, vocab_file='./vocab.pkl',
               start_word="<start>", end_word="<end>", unk_word="<unk>", vocab_from_file=True, num_workers=0,
               cocoapi_loc=r'D:/WatchAndTellCuda/'):
    """
    This function returns a torch.utils.data.DataLoader for custom coco dataset.
    :param transform: Image transform
    :param mode: either 'train' or 'test'
    :param batch_size: batch size (if in testing mode, must have batch_size=1)
    :param vocab_threshold: minimum word count threshold
    :param vocab_file: file containing the vocabulary
    :param start_word: special word denoting sentence start
    :param end_word: special word denoting sentence end
    :param unk_word: special word denoting unknown words
    :param vocab_from_file: if False, create vocab from scratch & override any existing vocab_file;
     if True, load vocab from existing vocab_file, if it exists
    :param num_workers: number of subprocesses to use for data loading
    :param cocoapi_loc: The location of the folder containing the COCO API
    """

    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    if not vocab_from_file:
        assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file
    if mode == 'train':
        if vocab_from_file:
            assert os.path.exists(
                vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'coco\\images\\train2017')
        annotations_file = os.path.join(cocoapi_loc, 'coco\\annotations\\captions_train2017.json')

    if mode == 'test':
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'coco\\images\\test2017')
        annotations_file = os.path.join(cocoapi_loc, 'coco\\annotations\\image_info_test2017.json')

    # COCO caption dataset
    dataset = CocoDataset(transform=transform, mode=mode, batch_size=batch_size, word_threshold=vocab_threshold,
                          vocab_file=vocab_file, start_word=start_word, end_word=end_word, unk_word=unk_word,
                          vocab_from_file=vocab_from_file, img_folder=img_folder, annotations_file=annotations_file)

    if mode == 'train':
        indices = dataset.get_train_indices()  # get indices for train/validation split
        init_sampler = data.sampler.SubsetRandomSampler(indices=indices)  # sampler for obtaining training batches
        data_loader = data.DataLoader(dataset=dataset, num_workers=num_workers,  # create data loader
                                      batch_sampler=data.sampler.BatchSampler(sampler=init_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=dataset.batch_size, shuffle=False,
                                      num_workers=num_workers)  # create data loader

    return data_loader


class CocoDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, word_threshold, vocab_file, start_word,
                 end_word, unk_word, vocab_from_file, img_folder, annotations_file):
        """
        This function initializes the dataset.
        :param transform: Image transform.
        :param mode: either 'train' or 'test'.
        :param batch_size: if in testing mode, must have batch_size=1.
        :param word_threshold: minimum word count threshold.
        :param vocab_file: file containing the vocabulary.
        :param start_word: special word denoting sentence start.
        :param end_word: special word denoting sentence end.
        :param unk_word: special word denoting unknown words.
        :param vocab_from_file: boolean. If False, create vocab from scratch & override any existing vocab_file.
        :param img_folder: folder containing images.
        :param annotations_file: coco format annotation file for images.
        """
        self.transform = transform  # image transformer
        self.mode = mode  # train or test
        self.batch_size = batch_size  # batch size
        self.vocab = Vocabulary(word_threshold, vocab_file, start_word, end_word,  # vocabulary wrapper
                                unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder  # folder with all the images

        if self.mode == 'train':  # training dataset
            self.coco = COCO(annotations_file)  # initialize COCO api for caption annotations
            self.ids = list(self.coco.anns.keys())  # obtain image ids
            print('Getting caption lengths...')  # get caption lengths
            all_tokens = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]['caption']).lower()
            ) for index in tqdm(np.arange(len(self.ids)))]  # tokenize captions
            self.caption_lengths = [len(token) for token in all_tokens]  # caption lengths
        else:
            test_info = json.load(open(annotations_file, 'r'))  # load json file
            self.paths = [item['file_name'] for item in test_info['images']]  # obtain image paths

    def __getitem__(self, index):

        if self.mode == 'train':  # training dataset
            coco = self.coco  # COCO API
            vocab = self.vocab  # vocabulary wrapper
            ann_id = self.ids[index]  # annotation id
            caption = coco.anns[ann_id]['caption']  # caption
            img_id = coco.anns[ann_id]['image_id']  # image id
            path = coco.loadImgs(img_id)[0]['file_name']  # image file name
            img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')  # load image

            if self.transform is not None:
                img = self.transform(img)  # apply image transformer

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())  # tokenize caption
            caption = [self.vocab(self.vocab.start_word)]  # start word
            caption.extend([self.vocab(token) for token in tokens])  # caption with start word
            caption.append(self.vocab(self.vocab.end_word))  # end word
            target = torch.Tensor(caption)  # convert caption to tensor
            lengths = len(caption)  # caption length
            lengths = torch.Tensor([lengths])  # convert caption length to tensor
            return img, target, lengths

        else:
            path = self.paths[index]  # image file name
            PIL_img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')  # load image
            original_image = np.array(PIL_img)  # convert image to numpy array

            if self.transform is not None:
                img = self.transform(PIL_img)  # apply image transformer
            return original_image, img

    def get_train_indices(self):
        """
        This function gets the indices for the training and validation set.
        :return: indices for training and validation set.
        """
        sel_lengths = np.random.choice(self.caption_lengths)  # select caption length
        all_indices = np.where([self.caption_lengths[i] == sel_lengths for i in range(len(self.caption_lengths))])[0]

        try:
            indices = np.random.choice(all_indices, self.batch_size, replace=False)
        except ValueError:  # if batch size is greater than caption length
            indices = np.random.choice(all_indices, self.batch_size, replace=True)
        return indices

    def __len__(self):
        return len(self.ids) if self.mode == 'train' else len(self.paths)  # return dataset size
