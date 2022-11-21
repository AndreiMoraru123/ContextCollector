import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):

    """
    The encoder is a pretrained ResNet-152 model that outputs a feature vector of size 2048 for each image.
    The output of the encoder is a tensor of shape (batch_size, 2048, encoded_image_size, encoded_image_size).
    :param encoded_image_size: The size of the encoded image.
    :return: A ResNet-152 model.
    """

    def __init__(self, encoded_image_size=28):
        super(EncoderCNN, self).__init__()
        self.encoded_image_size = encoded_image_size
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)  # to try: wide_resnet101_2
        modules = list(resnet.children())[:-2]  # remove avg-pool and fc layers
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_image_size, self.encoded_image_size))
        self.freeze_grad()  # freeze all the parameters but the last few layers

    def forward(self, images):
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def allow_grad(self):
        for p in self.resnet.parameters():
            p.requires_grad = True

    def freeze_grad(self):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True


class Attention(nn.Module):
    """
    Attention equation: alpha = softmax((W1 * h) + (W2 * s))
    where h is the output of the encoder, s is the hidden previous state of the decoder,
    and W1 and W2 are trainable weight matrices.

    :param encoder_out: output of the encoder (batch_size, num_pixels, encoder_dim)
    :param decoder_hidden: previous hidden state of the decoder (batch_size, decoder_dim)
    :return: attention weighted encoding, weights (batch_size, encoder_dim), (batch_size, num_pixels)
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()  # activation function
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):

    """
    The decoder is a LSTM that takes as input the encoded image and the previous word.
    :param attention_dim: The dimension of the attention network.
    :param embed_dim: The dimension of the word embeddings.
    :param decoder_dim: The dimension of the decoder LSTM.
    :param vocab_size: The size of the vocabulary.
    :param encoder_dim: The dimension of the encoder.
    :param dropout: The dropout rate.
    :return: A LSTM decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim  # the number of features in the encoded images
        self.attention_dim = attention_dim  # the dimension of the attention network
        self.embed_dim = embed_dim  # the dimension of the word embeddings
        self.decoder_dim = decoder_dim  # the dimension of the decoder RNN
        self.vocab_size = vocab_size  # the size of the vocabulary
        self.dropout = dropout  # dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)  # dropout layer
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()  # sigmoid layer to calculate attention gate
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)  # initialize linear layer with bias = 0
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  # get the mean because of adaptive average pooling
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)  # get the batch size
        encoder_dim = encoder_out.size(-1)  # get the feature size of the encoded images
        vocab_size = self.vocab_size  # get the vocabulary size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # flatten image
        num_pixels = encoder_out.size(1)  # get the number of pixels

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]  # sort the encoded images
        encoded_captions = encoded_captions[sort_ind].type(torch.LongTensor).to(device)

        embeddings = self.embedding(encoded_captions)  # embeddings (batch_size, max_caption_length, embed_dim)

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        decode_lengths = (caption_lengths - 1).tolist()  # no decoding at the <end> position (last position)
        decode_lengths = [int(i) for i in decode_lengths]  # convert to int because of error in pytorch

        # Create tensors to hold word prediciton scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])  # this is the batch size at time-step t
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding  # (batch_size_t, encoder_dim)
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds  # these are the predictions for the current time-step
            alphas[:batch_size_t, t, :] = alpha  # these are the attention weights for the current time-step

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def sample(self, encoder_out, data_loader, k=10):

        """
        Samples captions for given image features (Greedy search).
        :param encoder_out: the features extracted from the encoder
        :param data_loader: the data loader in which the vocabulary is stored
        :param k: the number of samples taken into account for the beam search
        :return: A list of sampled captions for the given image features and their corresponding attention weights
        """

        with torch.no_grad():

            enc_image_size = encoder_out.size(1)  # get the size of the encoded image
            encoder_dim = encoder_out.size(3)  # get the number of features in the encoded image
            vocab_size = self.vocab_size  # get the size of the vocabulary

            # Flatten image
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)  # get the number of pixels

            # We'll treat the problem as having a batch size of 1
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[data_loader.dataset.vocab('<start>')]] * k).to(device)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
                device)  # (k, 1, enc_image_size, enc_image_size)

            # Lists to store completed sequences, their alphas and scores
            complete_seqs = list()
            complete_seqs_alpha = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = self.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                torch.cuda.empty_cache()

                embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

                gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe  # this is the attention weighted encoding

                h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = self.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add the scores to the scores for the previous words
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                                       dim=1)  # (s, step+1, enc_image_size, enc_image_size)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != data_loader.dataset.vocab('<end>')]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())  # (s, step+1)
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])  # (s)
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]  # (s, step+1)
                seqs_alpha = seqs_alpha[incomplete_inds]  # (s, step+1, enc_image_size, enc_image_size)
                h = h[prev_word_inds[incomplete_inds].long()]  # (s, decoder_dim)
                c = c[prev_word_inds[incomplete_inds].long()]  # (s, decoder_dim)
                encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]  # (s, num_pixels, encoder_dim)
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)  # (s, 1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)  # (s, 1)

                # Break if things have been going on too long
                if step > 15:
                    break
                step += 1

            # if complete_seqs is empty, then we have to add the last incomplete sequence
            if len(complete_seqs) == 0:
                complete_seqs.extend(seqs.tolist())  # (s, step+1)
                complete_seqs_alpha.extend(seqs_alpha.tolist())  # (s, step+1, enc_image_size, enc_image_size)
                complete_seqs_scores.extend(top_k_scores.tolist())  # (s)

            i = complete_seqs_scores.index(max(complete_seqs_scores))  # (s)
            seq = complete_seqs[i]  # (s)
            alphas = complete_seqs_alpha[i]  # (s, enc_image_size, enc_image_size)

        return seq, alphas
