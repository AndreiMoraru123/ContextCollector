# COCO Context Collector

## It's a Contextualizer, trained on COCO! See what I did there?

### This mixed vision-language model gets better by making mistakes

![p1](https://user-images.githubusercontent.com/81184255/203029962-e26562e9-754d-4629-8330-b54e202698f2.gif)

This is the improved version of the original [Watch&Tell project](https://github.com/AndreiMoraru123/Watch-and-Tell). 

Check out [the old repo](https://github.com/AndreiMoraru123/Watch-and-Tell) to see what was improved and how to deal with the [Microsoft COCO dataset](https://cocodataset.org/#home).

Trained on the supervised 2017 challenge of image-caption pairs, using a custom data split and vocabulary generator.

Built using PyTorch :fire:

Explained in depth further down in this ```README```.

[Click here to see some more examples](#some-more-examples)

![p2](https://user-images.githubusercontent.com/81184255/203030363-57e342d2-6d74-4600-ae86-c09079f8ff22.gif)

#### Based on the original paper: [___Show, Attend and Tell___](https://arxiv.org/abs/1502.03044)

Frame goes in, caption comes out.

<p align="center">
  <img src="https://user-images.githubusercontent.com/81184255/203052548-e60eccde-59d9-48d5-a142-b32e5a24ccb7.png" width="500"/>
</p>

![p3](https://user-images.githubusercontent.com/81184255/203030392-61463981-f6bd-4921-85f0-d0b722584dba.gif)

## Motivation

The functional purpose of this project could be summed up as *Instance Captioning*, as in not trying to caption the whole frame, but only parts of it. This approach is not only going to be faster (because the model is not attempting to encode the information of the whole image), but it can also prove more reliable for video inference, through a very simple mechanism I will call "expansion". 

The deeper motivation for working on this is, however, more profound.

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

and, because the inference depends on the built vocabulary, the ```word_treshold``` can be set only while in ```training``` mode, and the ```vocab_from_file``` trigger can only be set to ```True``` while in ```testing``` mode.

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

Since ResNet was originally designed as a classifier, the last layer is going to be the activation function ```Softmax```. 

However, since PyTorch deals with probabilities implicitly using ```CrossEntropyLoss```, the classifier will not be present, and the only layers that need to be beheaded are the last linear fully connected layer and the average pooling layer, which will be replaced by the custom average pooling layer, for which you and I can choose the pooling size. 

The ```freeze_grad``` function is there if you need to tailor how many (if any) of the encoder layers do you want to train (optional, since the Net is pretrained).

The purpose of the resulting feature map is to provide a latent space representation of each frame, from which the decoder can draw multiple conclusions.

Any ResNet architecture (any depth) will work here, as well as some of the other predating CNNs (the paper used VGG), but keep in mind memory constraints for inference.

![p6](https://user-images.githubusercontent.com/81184255/203031528-ef8f6f19-f370-4372-9876-ce70f0e45731.gif)

# Attention

## Why?

"One important property of human perception is that one does not tend to process a whole scene
in its entirety at once. Instead humans focus attention selectively on parts of the visual space to
acquire information when and where it is needed" -- <cite>[___Recurrent Models of Visual Attention___](https://arxiv.org/abs/1406.6247) </cite>

The great gain of using attention as a mechanism in the decoder is that the importantce of the information contained in the encoded latent space is held into account and weighted (as in across all pixels of the latent space). Namely, the attention lifts the burden of having a single dominant state taking guesses about what is the context of information taken from the features by the model. The results are actually quite astounding when compared to an attention-less network (see previous project). 

## Where?

Since the encoder is already trained and can output a competent feature map (we know that ResNet can classify images), the mechanism of attention is used to augument the behaviour of the RNN decoder. During the training phase, the decoder learns which parts of the latent space make up the "context" of an image. The selling point of this approach is based on the fact that the learning is not done in a simple, sequential manner, but some non-linear interpolations can occur in such a way that you could make a strong point for convincing someone that the model has actually "understood" the task.

## What kind?

The original paper, as well as this implementation use [___Additive / Bahdanau Attention___](https://arxiv.org/abs/1409.0473)

The formula for the Bahdanau Attention is the essentially the following:

```
alpha = softmax((W1 * h) + (W2 * s))
```

where ```h``` is the output of the encoder, ```s``` is the hidden previous state of the decoder, and ```W1``` and ```W2``` are trainable weight matrices, producing a single number. (Note that the original paper also used ```tanh``` as a preactivation before ```softmax```. This implementation instead uses ```ReLU```.

:point_right: Additive attention is a model in and of itself, because it is in essence just a feed forward neural network. This is why it is built as an ```nn.Module``` class and inherits a forward call.

![p7](https://user-images.githubusercontent.com/81184255/203031544-2e57b5fd-44fd-4dc8-91c2-526ff7bc63da.gif)

# Decoder

I am using pretty much the same decoder proposed in the greatly elaborated [Image Captioning repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) with some caveats. Precisely:

1. I do not use padded sequences for the captions
2. I tailored tensor dimensions and types for a different pipeline (and dataset as well, the repo uses COCO 2014), so you may see differences
3. I am more lax with using incomplete captions in the beam search and I am also not concerned with visualizing the attention weights

The aformentioned implementation is self sufficient, but I will further explain how the decoder works for the purpose of this particular project, as well as the statements above.

The main idea of the model workflow is that the Encoder is passing a "context" feature to the decoder, which in turn produces an output. Since the decoder is an RNN, so the outputs will be given in sequences. The recurrent network can take into account the inputed features as well as its own hidden state.

The attention weighted encoding is gated through a sigmoid activation and the resulting values are added to the embedding of the previous word. This concatenation is then passed as the input to an ```LSTMCell```, along with the previous hidden state.

![p8](https://user-images.githubusercontent.com/81184255/203031558-6a519ad9-dd08-4fcf-ad0d-adf99c4c9740.gif)


## The LSTM Cell

The embedded image captions are concatenated with gated attention encodings and passed as the input of the LSTMCell. If this were an attentionless mechanism, you would just pass the encoded features added to the embeddings. 

Concatenation in code will look like this:

```
self.lstm = nn.LSTMCell(embeddings_size + encoded_features_size, decoded_hidden_size)  
```

The decoded dimension, i.e. the hidden size of the LSTMCell is obtained by concatennating the hidden an cell states.


```
hidden_state, cell_state = self.lstm( torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),  # input
                                      (hidden_state[:batch_size_t], cell_state[:batch_size_t]) )  # hidden
```

The cell outputs a tuple made out of the next hidden and cell states like in the picture down below. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/81184255/203302465-854077bf-ec2a-4cf7-9eaa-4f3621cf4d85.jpg" width = "500"/>
</p>


The intuition and computation behind the mechanism of the long short term memory unit are as follow:

The cell operates with a ___long term memory___ and a ___short term___ one. As their names intuitively convey, the former is concerned with a more general sense of state, while the latter is concentrated around what it has just seen. 

In the picture up above as well as in this model, ```h``` represents the ___short term memory___, or the  ___hidden state___, while ```c``` represents the ___long term memory___, or the ___cell state___.

1. The long term memory is initially passed through a ___forget gate___.The forget factor of this gate is computed using a ```sigmoid```, which ideally behaves like a binary selector (something either gets forgotten [0] or not [1]. In practice, most values will not be saturated so the information will be _somewhat_ forgotten (0,1). The current ___hidden state___ or ___short term memory___ is passed through the sigmoid to achieve this forget factor, which is then point-by-point multiplied with the ___long term memory___ or ___cell state___.
2. The short term memory will be joined by the ___input event___, ```x``` (which represents what the cell has just seen/experienced) in the ___input gate___, also called the ___learn gate___. This computation is done by gating both the input and the hidden state through an ___ignore gate___. The ignore factor of the gate is represented by a ```sigmoid``` to again ideally classify what has to be ignored [0] and what not [1]. How much is to be ignored is then decided by a ```tanh``` activation.
3. The ___long term memory___ joined by the newly aquired information in the ___input gate___ is passed into the ___remember gate___ and it becomes the new ___cell state___ and the new ___long term memory___ of the LSTM. The operation is a point-by-point addition of the two.
4. The ___output gate___ takes in all of the information from the input, hidden and cell state and becomes the new ___hidden state___ and ___short term memory___ of the network. The ___long term memory___ is passed through a ```tanh``` while the ___short term memory___ is passed through a ```sigmoid```, before being multiplied point-by-point in the final computation.

![p9](https://user-images.githubusercontent.com/81184255/203031581-b1dfb252-80af-438c-8353-04e04e649ed4.gif)

## Training the model

To train this model run the ```train.py``` file with the argument parsers tailored to your choice. My configuration so far has been something like this:

```
embed_size = 300  # this is the size of the embedding of a word, 
                  # i.e. exactly how many numbers will represent each word in the vocabulary.
                  # This is done using a look-up table through nn.Embedding 

attention_dim = 300  # this is the size of the full length attention dimension,
                     # i.e. exactly how many pixels are worth attenting to. 
                     # The pixels themselves will be learned through training
                     # and this last linear dimension will be sotfmax-ed 
                     # such as to output probabilities in the forward pass.

decoder_dim = 300  # this is the dimension of the hidden size of the LSTM cell
                   # and it will be the last input of the last fully connected layer
                   # that maps the vectorized words to their scores 
```

Now, there is no reason to keep all three at the same size, but you can intuitively see that it makes sense to keep them around the same range. You can try larger dimnesions, but keep in mind again hardware limitations, as these are held in memory.

The rest of the parsed arguments are:

```
dropout = 0.5  # the only drop out is at the last fully connected layer in the decoder,
               # the one that outputs the predictions based on the resulted hidden state of the LSTM cell
               
num_epochs = 5  # keep in mind that training an epoch may take several hours on most machines

batch_size = 22  # this one is as well depended on how many images can your GPU hold at once
                 # I cannot go much higher, so the training will take longer

word_threshold = 6  #  the minimum number of apparitions for a word to be included in the vocabulary

vocab_from_file = False  # if this is the first time of training / you do not have the pickle file,
                         # then you will have to generate the vocabulary first
                       
# save_every = 1  # save every chosen epoch

# print_every = 100  # log stats every chosen number of batches
```

The `loss` function is ```CrossEntropyLoss``` and should not be changed as this is the only one that makes sense. Captioning is just multi-label classifcation. 

The ```train_transform``` the images go through before being passed to the encoder is pretty standard, using the ImagNet ```mean``` and ```std``` values. 

Since the input sizes here do not vary it may make sense to set:

```
torch.backends.cudnn.benchmark = True  # optimize hardware algorithm
```

![p10](https://user-images.githubusercontent.com/81184255/203031587-69629719-fc88-4c1b-8ce5-76dc9b89aa36.gif)

## Beam Search

In the ```sample``` function of the decoder, there is an input parameter called ```k```. This one represents the number of captions held into consideration for future exploration. 

The beam search is a thing in machine translation, because you do not always want the next ___best___ word, as the word that comes after that may not be the ___overall best___ to form a meaningful sentence. 

Always looking for the next best is called a ___greedy___ search, and you can achieve that by setting ```k = 1```, such as to only hold one hypothesis every time. 

Again, keep in mind that, provided you have one, this search will also be transfered to your graphics card, so you may run out of memory if you try to keep count of too many posibilities. 

:point_right: That means you may sometimes be forced to either use a greedy search, or break the sentences before they finish.

I'll leave you with [this visual example](https://www.amazon.science/blog/amazon-open-sources-library-for-prediction-over-large-output-spaces) on how beam search can select two nodes in a graph instead of choosing only one.

<p align="center">
  <img src="https://user-images.githubusercontent.com/81184255/203261229-23030756-3b04-45cb-953e-dc819977961c.gif" width = "500"/>
</p>

![p11](https://user-images.githubusercontent.com/81184255/203032112-6fd1cef8-1768-4ea8-af16-068e89c3a302.gif)

## YOLO and the Perspective Expansion

Trying to output a caption for each frame of a video can be painful, even with attention. The model was trained on images from the COCO dataset, which are context rich scenarios, focused mainly on a single event, and thus  will perform as such on the testing set. 

But "real life" videos are different, each frame is related to the previous one and not all of them have much going on in one place, but rather many things happening at once.

* For this reason, I use [a tiny YOLOv4](https://github.com/AndreiMoraru123/ContextCollector/tree/main/YOLO) model to get an initial object of interest in the frame. 
* A caption is then generated for the region of interest (ROI) bounded by the YOLO generated box
* If the prediction is far off the truth (no word in the sentence matches the label output by the detector), the algo expands the ROI by a given factor until it does or until a certain number of tries have been made, to avoid infinite loops
* Using the newly expanded ROI, the model is able to get more context out of the frame
* As you can see in the examples, the expansion factor usually finds its comfortable space before reaching a full sized image
* That means there are significant gains in inference speeds and better predictions
* Much like in [Viola Jones](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf), this model expands, but not when being correct.
* Instead, it grows by making obvious mistakes, and in fact relies on it to give its best performance in terms of context understanding.

![p12](https://user-images.githubusercontent.com/81184255/203032117-f7f80c93-ffea-46f4-a282-37195384f4b3.gif)

## Inference Pipeline

I provided some model ```pruning``` functions in the ```pipeline.py``` file, both structured and unstructured (global and local), but I use neither and do not recommend them as they are now. You could achieve faster inference by cutting out neurons or connections, but you will also hinder the performance.

I highly avoid structured pruning (both L1 and L2), as it will just wipe out most of the learned vocabulary, at no speed gains.

Example:

```
a man <unk> <unk> <unk> a <unk> <unk> <unk> <unk> .
a man <unk> <unk> <unk> a <unk> <unk> <unk> .
a <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> .
a <unk> <unk> <unk> <unk> <unk> <unk> <unk> .
```

While unstructured (both local and global) pruning is safer:

```
a man on a motorcycle in the grass .
a motorcycle parked on the side of the road .
a man on a skateboard in a park .
a person on a motorcycle in the woods .
```

But no more performant in terms of speed

Local pruning works layer by layer across every layer, while global pruning wipes across all layers indiscriminately. But for the purpose of this model, they both produce no gain.

Unstructured pruning is always L1, because the weights are sorted one after the other.

the ```JIT``` compiler can be used to increase the performance using the ```optimized_execution```. However, this does not always result in a smaller model, and it could in fact make the network increase in size. 

Neither ```torch.jit``` nor ```onnx``` converters can be used on the decoder, because it is very customized, and these operations for now require strong tensor typing, and are not very permissive to custom architectures, so I resorted to only tracing the ResNet encoder (which also cannot be inferenced using ```onnxruntime```, because of the custom average pooling layer). 

As you can start to see, there are not really any out of the box solutions for these types of things yet.

The rest of the inference pipeline just loads the ```state_dicts``` of each model and runs the data stream through them using a pretty standard ```test_transform``` and dealing with the expansion of the ROI.

![p13](https://user-images.githubusercontent.com/81184255/203032125-af4328cd-4ff2-4eb2-a66d-61807fbbb925.gif)

## Running the model

To test the model you can run the ```run.py``` file by parsing the needed arguments.

Since the inference of the model relies on teacher forcing, i.e. using the whole caption for inference regardless of the last generated sequence, for now the whole vocabulary is needed to test the model, meaning that the dataset is needed as well, so one will have to train the model before running it the way it works now. 

I also cannot provide the encoder here as there are size constraints, but any pretrained resnet will work (do make sure to behead it first if you choose to try this out).

The options for running the model are as follow:

```
--video  # this is an mp4 video that will be used for inference, I provide one in the video folder
--expand  # this is the expanding ratio of the bounding box ROI after each mistake
--backend  # this is best set to 'cuda', but be weary of memory limitations
--k  # this is the number of nodes (captions) held for future consideration in the beam search
--conf  # this is the confidence threshold for YOLO
--nms  # this is the non-maximum suppression for the YOLO rendered bounding boxes
```

YOLO inference is done using the ```dnn``` module from ```OpenCV```.


![p14](https://user-images.githubusercontent.com/81184255/203032384-3a2cb769-bc94-45e0-9048-e6eecbe75fd6.gif)

## Hardware and Limitations

My configuration is the following:

<p align="center">
  <img src="https://user-images.githubusercontent.com/81184255/203286519-82ed38a6-d63a-424b-bc91-bf906412bf68.png" width = "500"/>
</p>

I am using:
* a turing Geforce GTX 1660 TI with 6GB of memory (CUDA arch bin of 7.5)
* CUDA 11.7
* cuDNN 8.5 (so that it works with OpenCV 4.5.2) 

:point_right: Be aware that when building OpenCV there will be no errors if your pick incompatible versions, but unless everything clicks the net will refuse to run of the GPU

Using the computation ```FPS = 1 / inference_time```, the model is able to average 4 frames per second.

![p15](https://user-images.githubusercontent.com/81184255/203032605-d671478d-c46f-4292-9727-6bcd74dd724c.gif)

## Future outlook and goals

What I am currently looking into is optimization. 

The current model is working, but in a hindered state. With greater embeddings and a richer vocabulary the outputs can potentially be better. Training in larger batches will also finish faster. 

For this reason, I am now currently working on Weight Quantization and Knowledge Distillation.

I am also currently looking into deployment tools using ONNX. 

These are both not provided off the bat for artificial intelligence models, so there is really no go-to solution. I will keep updating the repository as I make progress.

I am also playing around with the [Intel Neural Compute Stick](https://www.intel.com/content/www/us/en/developer/tools/neural-compute-stick/overview.html) and the OpenVINO api to split the inference of the different networks away from running out of CUDA memory.

![p16](https://user-images.githubusercontent.com/81184255/203032623-d02fb14a-8054-421e-9bba-306784d91207.gif)

# Some more examples

Notice how in the motorcycle example the ROI expands until it can notice there is not only one, but a group of people riding motorcycles, something object detection itself is incapable of accomplishing. 

Shift                    |           In             |         Perspective
:-------------------------:|:-------------------------:|:-------------------------:
![p1m](https://user-images.githubusercontent.com/81184255/203171977-04178a2a-0b08-4114-af66-eb65662cf578.gif) | ![p2m](https://user-images.githubusercontent.com/81184255/203172037-2e26db28-9745-4f12-88f1-b6a1b8143356.gif) | ![p3m](https://user-images.githubusercontent.com/81184255/203172063-cadc682a-529d-413d-a7d5-343bbd66af71.gif)

![p1](https://user-images.githubusercontent.com/81184255/203173602-62e9234d-5043-47fd-8bcb-6942017a0de2.gif)

Broaden                    |           The             |         View
:-------------------------:|:-------------------------:|:-------------------------:
![p1](https://user-images.githubusercontent.com/81184255/203182932-454712e1-a2ce-4bc4-91b6-3a5103944160.gif) | ![p2](https://user-images.githubusercontent.com/81184255/203182945-4e37635b-88e5-4f3e-acea-b9dec795d2a9.gif) | ![p3](https://user-images.githubusercontent.com/81184255/203182986-836c4610-d8a0-4043-abbf-c7eee78fb5ed.gif)

```bibtex
@misc{https://doi.org/10.48550/arxiv.1502.03044,
  doi = {10.48550/ARXIV.1502.03044},
  url = {https://arxiv.org/abs/1502.03044},
  author = {Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
  publisher = {arXiv},
  year = {2015},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
