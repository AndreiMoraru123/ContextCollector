from pycocotools.coco import COCO
from cocodata import get_loader
from model import EncoderCNN, DecoderRNN, device
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import sys
import math
import argparse

sys.path.append('PythonAPI')
dataDir = r'D:/WatchAndTellCuda/'
dataType = 'train2017'
annFile = '{}coco/annotations/instances_{}.json'.format(dataDir, dataType)
log_file = 'training_log.txt'

# initialize COCO api
coco = COCO(annFile)
torch.cuda.empty_cache()

cudnn.benchmark = True  # enable benchmarking optimization in cudnn

transform_train = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # ImageNet params
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


def train(embed_size=300, attention_dim=300, decoder_dim=300, dropout=0.5, num_epochs=5,
          batch_size=22, word_threshold=6, vocab_from_file=True, save_every=1, print_every=100):

    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=word_threshold,
                             vocab_from_file=vocab_from_file)

    vocab_size = len(data_loader.dataset.vocab)

    encoder = EncoderCNN()
    decoder = DecoderRNN(attention_dim, embed_size, decoder_dim, vocab_size, dropout=dropout)

    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    params = list(decoder.parameters())  # only train the decoder params
    optimizer = torch.optim.Adam(params, lr=0.001)
    # total number of training steps
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
    print('Training on:', device)

    with open(log_file, 'w') as f:

        for epoch in range(1, num_epochs + 1):

            for i_step in tqdm(range(1, total_step + 1)):

                indices = data_loader.dataset.get_train_indices()  # get random indices
                sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)  # create a sampler
                data_loader.batch_sampler.sampler = sampler  # assign the sampler to the batch sampler

                images, captions, lengths = next(iter(data_loader))  # get the next batch
                images = images.to(device)
                captions = captions.to(device)
                lengths = lengths.to(device)

                features = encoder(images)  # extract features
                outputs, caps_sorted, decode_lengths, alphas, sort_indices = decoder(features, captions, lengths)

                targets = caps_sorted[:, 1:]  # remove <start> token

                optimizer.zero_grad()  # zero the gradients

                loss = criterion(outputs.view(-1, vocab_size), targets.contiguous().view(-1))  # calculate loss
                loss.backward()  # back-propagate

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.1)  # clip gradients

                optimizer.step()  # update weights

                if i_step % print_every == 0:
                    f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} \n'
                            .format(epoch, num_epochs, i_step, total_step, loss.item()))

            if epoch % save_every == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    'models', 'decoder-{}-300.ckpt'.format(epoch)))
                torch.save(encoder.state_dict(), os.path.join(
                    'models', 'encoder-{}-300.ckpt'.format(epoch)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--attention_dim', type=int, default=300, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int, default=300, help='dimension of decoder RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=22, help='batch size, important for memory')
    parser.add_argument('--word_threshold', type=int, default=6, help='minimum word count threshold')
    parser.add_argument('--vocab_from_file', type=bool, default=True,
                        help='if True, load existing vocab file. If False, create vocab file from scratch')
    parser.add_argument('--save_every', type=int, default=1, help='save every n epochs')
    parser.add_argument('--print_every', type=int, default=100, help='print every n steps')

    args = parser.parse_args()

    train(embed_size=args.embed_size, attention_dim=args.attention_dim, decoder_dim=args.decoder_dim,
          dropout=args.dropout, num_epochs=args.num_epochs, batch_size=args.batch_size,
          word_threshold=args.word_threshold, vocab_from_file=args.vocab_from_file,
          save_every=args.save_every, print_every=args.print_every)
