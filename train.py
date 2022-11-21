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

sys.path.append('PythonAPI')
dataDir = r'D:/WatchAndTellCuda/'
dataType = 'train2017'
annFile = '{}coco/annotations/instances_{}.json'.format(dataDir, dataType)
log_file = 'training_log.txt'

# initialize COCO api
coco = COCO(annFile)
torch.cuda.empty_cache()

embed_size = 300
attention_dim = 300
decoder_dim = 300
dropout = 0.5

cudnn.benchmark = True  # enable benchmarking optimization in cudnn

batch_size = 22
word_threshold = 6  # minimum word count threshold (i.e. if a word occurs less than 6 times, it is discarded)
vocab_from_file = True  # if True, load existing vocab file
num_epochs = 3
save_every = 1  # save every 1 epochs
print_every = 100  # print training/validation stats every 100 batches

transform_train = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # ImageNet params
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

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


if __name__ == '__main__':

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
