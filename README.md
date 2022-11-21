# COCO Context Collector

## It's a COntextualizer, trained on COCO! See what I did there? :zany_face:

### This mixed vision-language model gets better by making mistakes

![p1](https://user-images.githubusercontent.com/81184255/203029962-e26562e9-754d-4629-8330-b54e202698f2.gif)

This is the improved version of the original [Watch&Tell project](https://github.com/AndreiMoraru123/Watch-and-Tell). 

Check out [the old repo](https://github.com/AndreiMoraru123/Watch-and-Tell) to see what was improved and how to deal with the [Microsoft COCO dataset](https://cocodataset.org/#home).

Trained on the supervised 2017 challenge of image-caption pairs, using a custom data split and vocabulary generator.

Built using PyTorch :fire:

Explained in depth further down in this ```README```.

![p2](https://user-images.githubusercontent.com/81184255/203030363-57e342d2-6d74-4600-ae86-c09079f8ff22.gif)

#### Based on the original paper: [___Show, Attend and Tell___](https://arxiv.org/abs/1502.03044)

Frame goes in, caption comes out.

<img src="https://user-images.githubusercontent.com/81184255/203052548-e60eccde-59d9-48d5-a142-b32e5a24ccb7.png" width="1000" height="500"/>

![p3](https://user-images.githubusercontent.com/81184255/203030392-61463981-f6bd-4921-85f0-d0b722584dba.gif)

## Motivation

The functional purpose of this project could be summed up as *Instance Captioning*, as in not trying to caption the whole frame, but only parts of it. This approach is not only going to be faster (because the model is not attempting to encode the information of the whole image), but it can also prove more reliable for video inference, through a very simple mechanism I will call "expansion". 

The deeper motivation for working on this is, however, more profound:

For decades, language and vision were treated as completely different problems and naturally, the paths of engineering that have emerged to provide solutions for them were divergent to begin with.

Neural networks, while perhaps the truce between the two, as their application in deep learning considerably improved both language and vision, still today rely mostly on different techniques for each task, as if language and vision would be disconnected from one another. 

The latest show in town, the Transformer architecture, has provided a great advancement into the world of language models, following the original paper [___Attention is All You Need___](https://arxiv.org/abs/1706.03762) that paved the way to models like [___GPT-3___](https://arxiv.org/abs/2005.14165), and while the success has not been completely transferred to vision, some breakthroughs have been made: [___An Image is Worth 16x16 Words___](https://arxiv.org/abs/2010.11929), [___SegFormer___](https://arxiv.org/abs/2105.15203), [___DINO___](https://arxiv.org/abs/2104.14294).

One of the very newest (time of writing: fall 2022) is [Google's LM-Nav](https://sites.google.com/view/lmnav?pli=1), a Large Vision + Language model used for robotic navigation. What is thought provoking about this project is the ability of a combined V+L model to "understand" the world better than a V or L model would do on their own. Perhaps human intelligence itself is the sum of smaller combined intelligent models. The robot is presented with conflicting scenarios and is able to even "tell" if a prompt makes sense as a navigational instruction or is impossible to fulfil.

![p4](https://user-images.githubusercontent.com/81184255/203030436-eb2e37bd-6b83-45bc-84a6-a527c52bb765.gif)

## Vocabulary and Data

As the official dataset homespage states, "COCO is a large-scale object detection, segmentation, and captioning dataset".

For this particular model, I am concerned with detection and captioning.

Before the ```CocoDataset``` can be created in the [cocodata.py](https://github.com/AndreiMoraru123/ContextCollector/blob/main/cocodata.py) file, a ```vocabulary``` instance of the ```Vocabulary``` class has to be constructed using the [vocabulary.py](https://github.com/AndreiMoraru123/ContextCollector/blob/main/vocabulary.py) file. This can be conveniently done using the ```tokenize``` function of of ```nltk``` module.

The Vocabulary is simply the collection of words that the model needs to learn. It also needs to convert said words into numbers, as the decoder can only process them as such. To be able to read the output of the model, they also need to be converted back. These two are done using two hash map structures (Python Dictionaries), ```word2idx``` and ```idx2word```.

As per all sequence to sequence models, the vocab has to have a known ```<start>``` token, as well as an ```<end>``` one. An ```<unk>``` token for the unknown words, yet to be added to the file acts as a selector for what gets in. 

The vocabulary is, of course, built on the COCO annotations available for the images.

:point_right: The important thing to know here is that each vocabulary generation can (and should) be customized. The instance will not simply add all the words that it can find in the annotations file, because a lot would be redundant. 

For this reason, two vocabulary "hyper-parameters" can be tuned:

```python
word_threshold = 6  # minimum word count threshold (i.e. if a word occurs less than 6 times, it is discarded)
vocab_from_file = False  # if True, load existing vocab file. If False, create vocab file from scratch
```

and, because the inference depends on the built vocabulary, the ```word_treshold``` can be set only while ```training``` mode, and the ```vocab_from_file``` trigger can only be set to ```True``` while in ```testing``` mode.

Building the vocabulary will generate the ```vocab.pkl``` pickle file, which can then be later loaded for inference.

![p5](https://user-images.githubusercontent.com/81184255/203030454-9c023413-e532-444f-9b97-ae4ee14034f1.gif)

## Model description

As found in the [model.py](https://github.com/AndreiMoraru123/ContextCollector/blob/main/model.py)

### 1. [The CNN Encoder](#encoder)
### 2. [The Attention Network](#attention)
### 3. [The RNN Decoder](#decoder)

<img align="left" src="https://user-images.githubusercontent.com/81184255/203086410-5f872451-1fbc-41a8-a624-3d8ebb11c35a.png" />

# Encoder

The encoder is a beheaded pretrained ResNet-152 model that outputs a feature vector of size 2048 x W x H  for each image, where W and H are both the ```encoded_image_size``` used in the last average pooling. The original paper proposed an encoded size of 14. 

As ResNet was originally designed as a classifier, the last layer is going to be the activation function ```Softmax```. 

However, since PyTorch deals with it using implicitly using ```CrossEntropyLoss```, the only layers that need to be beheaded are the last linear fully connected layer and the average pool layer, which will be replaced by the custom average pool layer, for which we you and I can choose the pooling size. 

The ```freeze_grad``` function is there if you need to tailor how many (if any) of the encoder layers do you want to train (optional, since the Net is pretrained).

The purpose of the resulting feature map is to provide a latent space representation of each frame, from which the decoder can draw multiple conclusions.

Any ResNet architecture (any depth) will work here, as well as some of the other predating CNNs (the paper used VGG), but keep in mind memory constraints for inference.

![p6](https://user-images.githubusercontent.com/81184255/203031528-ef8f6f19-f370-4372-9876-ce70f0e45731.gif)

# Attention

## Why?

"One important property of human perception is that one does not tend to process a whole scene
in its entirety at once. Instead humans focus attention selectively on parts of the visual space to
acquire information when and where it is needed" -- <cite>[___Recurrent Models of Visual Attention___](https://arxiv.org/abs/1406.6247) </cite>

The great gain of using attention as a mechanism in the decoder is that the importantce the information contained in the encoded latent space is held into account and weighted (as in across all pixels of the latent space. Namely, the attention lifts the burden of having a single dominant state taking guesses about what is the context of information taken from the decoder. The results are reallly actually astounding when compared to an attention-less network (see previous project). 

## Where?

Since the encoder is already trained and can output a competent feature map (we know that ResNet can classify images), the mechanism of attention is used to augument the behaviour of the RNN decoder. During the training phase, the decoder learns which parts of the latent space make up the "context" of an image. The selling point of this approach is based on the fact that the learning is not done in a simple, sequential manner, but some non-linear interpolations can occur in such a way that you could make a strong point for convincing someone that the model has actually "understood" the task.

## What kind?

The original paper, as well as this implementation use [___Additive / Bahdanau Attention___](https://arxiv.org/abs/1409.0473)

The formula for the Bahdanau Attention is the essentially the following:

```
alpha = softmax((W1 * h) + (W2 * s))
```

where h is the output of the encoder, s is the hidden previous state of the decoder, and W1 and W2 are trainable weight matrices, producing a single number. (Note that the original paper also used ```tanh``` as a preactivation before ```softmax```. This implementation instead uses ```ReLU```.

Additive attention is a model in and of itself, because it is in essence just a feed forward neural network. This is why it is built as an ```nn.Module``` class and inherits a forward call.

![p7](https://user-images.githubusercontent.com/81184255/203031544-2e57b5fd-44fd-4dc8-91c2-526ff7bc63da.gif)

# Decoder

I am using pretty much the same decoder proposed in the greatly elaborated [Image Captioning repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) with some caveats. Precisely:

1. I do not use padded sequences for the captions
2. I tailored tensor dimensions and types for a different pipeline (and dataset as well, the repo uses COCO 2014), so you may see differences
3. I am more lax with using incomplete captions in the beam search and I am also not concerned with visualizing the attention weights

The aformentioned implementation is self sufficient, but I will further explain how the decoder works for the purpose of this particular project, as well as the statements above.

The main idea of the model workflow is that the Encoder is passing a "context" feature to the decoder, which in turn produces an output. Since the decoder is an RNN, so the outputs will be given in sequences. The recurrent network can take into account the inputed features as well as it's own hidden state.

The attention weighted encoding is gated through a sigmoid activation and the resulting values are added to the embedding of the previous word. This concatenation is then passed as the input to an ```LSTMCell```, along with the previous hidden state.

![p8](https://user-images.githubusercontent.com/81184255/203031558-6a519ad9-dd08-4fcf-ad0d-adf99c4c9740.gif)


## The LSTM Cell

![lstm](https://user-images.githubusercontent.com/81184255/203153685-bdbb2818-541b-4844-8944-24993394af9b.jpg)

![p9](https://user-images.githubusercontent.com/81184255/203031581-b1dfb252-80af-438c-8353-04e04e649ed4.gif)

## Training the model

![p10](https://user-images.githubusercontent.com/81184255/203031587-69629719-fc88-4c1b-8ce5-76dc9b89aa36.gif)

## Beam Search

![p11](https://user-images.githubusercontent.com/81184255/203032112-6fd1cef8-1768-4ea8-af16-068e89c3a302.gif)

## YOLO and the Perspective Expansion

![p12](https://user-images.githubusercontent.com/81184255/203032117-f7f80c93-ffea-46f4-a282-37195384f4b3.gif)

## Inference Pipeline

![p13](https://user-images.githubusercontent.com/81184255/203032125-af4328cd-4ff2-4eb2-a66d-61807fbbb925.gif)

## Running the model

![p14](https://user-images.githubusercontent.com/81184255/203032384-3a2cb769-bc94-45e0-9048-e6eecbe75fd6.gif)

## Hardware and Limitations

![p15](https://user-images.githubusercontent.com/81184255/203032605-d671478d-c46f-4292-9727-6bcd74dd724c.gif)

## Optimization

![p16](https://user-images.githubusercontent.com/81184255/203032623-d02fb14a-8054-421e-9bba-306784d91207.gif)
