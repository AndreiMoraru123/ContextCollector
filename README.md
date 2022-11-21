# COCO Context Collector

## It's a COntextualizer, trained on COCO! See what I did there? :zany_face:

### This mixed vision-language model gets better by making mistakes

![p1](https://user-images.githubusercontent.com/81184255/203029962-e26562e9-754d-4629-8330-b54e202698f2.gif)

This is the improved version of the original [Watch&Tell project](https://github.com/AndreiMoraru123/Watch-and-Tell). 

Check out [the old repo](https://github.com/AndreiMoraru123/Watch-and-Tell) to see what was improved and how to deal with the [Microsoft COCO dataset](https://cocodataset.org/#home).

Trained on the supervised 2017 challenge of image-caption pairs, using a custom data split and vocabulary generator.

Explained in depth further down in this ```README```.

![p2](https://user-images.githubusercontent.com/81184255/203030363-57e342d2-6d74-4600-ae86-c09079f8ff22.gif)

#### Based on the original paper: [___Show, Attend and Tell___](https://arxiv.org/abs/1502.03044)

Frame goes in, caption comes out.

<img src="https://user-images.githubusercontent.com/81184255/203052548-e60eccde-59d9-48d5-a142-b32e5a24ccb7.png" width="1000" height="500"/>

![p3](https://user-images.githubusercontent.com/81184255/203030392-61463981-f6bd-4921-85f0-d0b722584dba.gif)

## The COCO Dataset

As the official dataset homespage states, "COCO is a large-scale object detection, segmentation, and captioning dataset".

The functional purpose of this project could be summed *Instance Captioning*, as in not trying to caption the whole frame, but only parts of it. This approach is not only going to be faster (because the model is not trying to encode the whole image), but can also prove more reliable for video inference, through a very simple mechanism I will call "expansion". 

The deeper motivation for working on this is, however, more profound:

For decades, language and vision were treated as completely different problems and naturally, the engineered paths that have emerged to provide solutions for them were divergent to begin with.

Neural networks, while perhaps the truce between the two, as their application in deep learning considerably improved both language and vision, still today rely mostly on different techniques for each task, as if language and vision would be disconnected from one another. 

The latest show in town, the Transformer architecture, has provided a great advancement into the world of language models, following the original paper [___Attention is All You Need__](https://arxiv.org/abs/1706.03762), and while the success has not been completely transferred to vision, some breakthroughs have been made [___An Image is Worth 16x16 Words___](https://arxiv.org/abs/2010.11929), [___SegFormer___](https://arxiv.org/abs/2105.15203), [___DINO___](https://arxiv.org/abs/2104.14294).

One of the very newest (time of writing: fall 2022) is [Google's LM-Nav](https://sites.google.com/view/lmnav?pli=1), a Large Vision + Language model used for robotic navigation. What is thought provoking about this project is the ability of a combined V+L model to "understand" the words better than a V or L model would do on their own. Perhaps human intelligence itself is the sum of smaller combined intelligent models. The robot is presented with conflicting scenarios and is able to even "tell" if a promp makes sense as a navigational instruction.

![p4](https://user-images.githubusercontent.com/81184255/203030436-eb2e37bd-6b83-45bc-84a6-a527c52bb765.gif)

## Vocabulary Builder

![p5](https://user-images.githubusercontent.com/81184255/203030454-9c023413-e532-444f-9b97-ae4ee14034f1.gif)

## Model description

### 1. [Encoder](##encoder)
### 2. Attention Network
### 3. Decoder

## Encoder

![p6](https://user-images.githubusercontent.com/81184255/203031528-ef8f6f19-f370-4372-9876-ce70f0e45731.gif)

![p7](https://user-images.githubusercontent.com/81184255/203031544-2e57b5fd-44fd-4dc8-91c2-526ff7bc63da.gif)

![p8](https://user-images.githubusercontent.com/81184255/203031558-6a519ad9-dd08-4fcf-ad0d-adf99c4c9740.gif)

![p9](https://user-images.githubusercontent.com/81184255/203031581-b1dfb252-80af-438c-8353-04e04e649ed4.gif)

![p10](https://user-images.githubusercontent.com/81184255/203031587-69629719-fc88-4c1b-8ce5-76dc9b89aa36.gif)

![p11](https://user-images.githubusercontent.com/81184255/203032112-6fd1cef8-1768-4ea8-af16-068e89c3a302.gif)

![p12](https://user-images.githubusercontent.com/81184255/203032117-f7f80c93-ffea-46f4-a282-37195384f4b3.gif)

![p13](https://user-images.githubusercontent.com/81184255/203032125-af4328cd-4ff2-4eb2-a66d-61807fbbb925.gif)

![p14](https://user-images.githubusercontent.com/81184255/203032384-3a2cb769-bc94-45e0-9048-e6eecbe75fd6.gif)

![p15](https://user-images.githubusercontent.com/81184255/203032605-d671478d-c46f-4292-9727-6bcd74dd724c.gif)

![p16](https://user-images.githubusercontent.com/81184255/203032623-d02fb14a-8054-421e-9bba-306784d91207.gif)
