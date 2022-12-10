from model import EncoderCNN, DecoderRNN, device
from torch.backends import cudnn
from cocodata import get_loader
from torchvision import transforms
import torch
from PIL import Image
import cv2
import os
import torch.nn.utils.prune as prune
import numpy as np

cudnn.benchmark = True

encoder_file = 'encoder-5-300.ckpt'
decoder_file = 'decoder-5-300.ckpt'

encoder = EncoderCNN()
encoder.load_state_dict(torch.load(os.path.join('models', encoder_file)))
encoder.eval()
encoder.to(device)

dummy_input = torch.randn(1, 3, 480, 480, device='cuda')
with torch.jit.optimized_execution(True):
    encoder = torch.jit.trace(encoder, dummy_input)
    encoder.save("models/encoder.pt")

encoder = torch.jit.load(os.path.join('models', 'encoder.pt'))

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

data_loader = get_loader(transform=transform_test,
                         mode='test')

embed_size = 300
attention_dim = 300
decoder_dim = 300

vocab_size = len(data_loader.dataset.vocab)
decoder = DecoderRNN(embed_size, attention_dim, decoder_dim, vocab_size, dropout=0.5)
decoder.load_state_dict(torch.load(os.path.join('models', decoder_file)))
decoder.eval()
decoder.to(device)


def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model


def prune_model_l1_structured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
    return model


def prune_model_l2_structured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, 'weight', proportion, n=2, dim=1)
            prune.remove(module, 'weight')
    return model


def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h, colors, classes, k):
    """
    Draw the predicted bounding box
    :param img: the frame
    :param class_id: the class id
    :param x: the x coordinate of the bounding box
    :param y: the y coordinate of the bounding box
    :param x_plus_w: the width of the bounding box
    :param y_plus_h: the height of the bounding box
    :param colors: the colors of the bounding box
    :param classes: the classes of the bounding box
    :return: the predicted sentence as both string and a list, and the predicted label
    """
    label = str(classes[class_id])
    color = colors[class_id]

    if label == 'motorbike' or label == 'car' or label == 'truck' or label == 'bus' or label == 'bicycle' \
            or label == "person" or label == "dog" or label == "cat" or label == "horse" or label == "bird" \
            or label == "cow":

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        roi = img[y: y_plus_h, x: x_plus_w]

        if roi.size != 0:

            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray((roi * 255).astype(np.uint8))
            roi = transform_test(roi)
            roi = roi.unsqueeze(0)
            roi = roi.to(device)

            features = encoder(roi)
            seq, _ = decoder.sample(features, data_loader, k=k)

            sentence = clean_sentence(seq)

            word_list = []

            for idx in seq:
                if idx != 0 and idx != 1:
                    word_list.append(data_loader.dataset.vocab.idx2word[idx])

            for i in range(len(word_list)):
                if word_list[i] == 'man' or word_list[i] == 'woman' or word_list[i] == 'girl' or word_list[i] == 'boy':
                    word_list[i] = 'person'
                if word_list[i] == 'motorcycle':
                    word_list[i] = 'motorbike'

            cv2.putText(img, sentence, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(sentence)

            return sentence, word_list, label


def clean_sentence(seq):
    return str(' '.join([data_loader.dataset.vocab.idx2word[idx] for idx in seq if (idx != 0 and idx != 1)]))
