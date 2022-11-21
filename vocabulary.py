import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter


class Vocabulary(object):

    def __init__(self, word_threshold, vocab_file='./vocab.pkl',
                 start_word="<start>", end_word="<end>", unk_word="<unk>",
                 annotations_file='..\\coco\\annotations\\captions_train2017.json',
                 vocab_from_file=False):

        """
          This class is used to build a vocabulary for the COCO dataset.
          :param word_threshold: Minimum word count threshold.
          :param vocab_file: File containing the vocabulary.
          :param start_word: Special word denoting sentence start.
          :param end_word: Special word denoting sentence end.
          :param unk_word: Special word denoting unknown words.
          :param annotations_file: Path for train annotation file.
          :param vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from existing vocab_file, if it exists
        """

        self.word_threshold = word_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)  # load the vocabulary from file
                self.word2idx = vocab.word2idx  # word to index
                self.idx2word = vocab.idx2word  # index to word
            print('Loaded vocabulary from file: ', self.vocab_file)
        else:
            self.build_vocab()  # build the vocabulary from scratch
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)  # save the vocabulary to file

    def build_vocab(self):
        """
        Populate the dictionaries for converting tokens to integers (and vice-versa).
        """
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """
        Initialize the dictionaries for converting tokens to integers (and vice-versa).
        """
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx  # add word to word2idx
            self.idx2word[self.idx] = word  # add word to idx2word
            self.idx += 1  # increment index

    def add_captions(self):
        """
        Add all captions to the vocabulary.
        """
        coco = COCO(self.annotations_file)  # initialize COCO api for caption annotations
        counter = Counter()  # count all the words
        ids = coco.anns.keys()  # get all the caption ids

        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])  # get the caption
            tokens = nltk.tokenize.word_tokenize(caption.lower())  # tokenize the caption
            counter.update(tokens)  # update the counter

            if i % 1000 == 0:
                print("[%d/%d] Tokenized the captions." %(i, len(ids)))  # print the progress

        # get all the words that occur more than threshold times
        words = [word for word, cnt in counter.items() if cnt >= self.word_threshold]

        for i, word in enumerate(words):
            self.add_word(word)  # add the words to the vocabulary

        print("Built vocabulary with %d words" %len(words))
        print("Vocabulary file saved to: ", self.vocab_file)

    def __call__(self, word):
        """
        If the word is present in the vocabulary, return its index.
        :param word: the word to be looked up.
        :return: the index of the word taken from the vocabulary dictionary.
        """
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        """
        :return: the length of the vocabulary.
        """
        return len(self.word2idx)
