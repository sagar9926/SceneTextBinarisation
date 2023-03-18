import os
import xml.etree.ElementTree as ET
import numpy as np
import keras_ocr


def extract_labels(xml):
    data = ET.parse(xml)
    root = data.getroot()
    img_name = []
    text = []
    for img in root.findall('image'):
        txt = ''
        img_name.append(img.find('imageName').text.replace('/', '_'))
        for rec in img.findall('taggedRectangles'):
            words = []
            for t in rec.findall('taggedRectangle'):
                words.append(t.find('tag').text)
            txt = ' '.join(words)
        text.append(txt)
    return np.array(img_name), np.array(text)


def extract_text_from_image(folder):
    text = []
    path = os.path.join(os.getcwd(), folder)
    img_path = []
    for img in os.listdir(path):
        img_path.append(os.path.join(path, img))
    imgs = []
    for img_pth in img_path:
        imgs.append(keras_ocr.tools.read(img_pth))
    pipeline = keras_ocr.pipeline.Pipeline()
    for img in imgs:
        pred = pipeline.recognize([img])[0]
        words = [tup[0] for tup in pred]
        text.append(" ".join(words))
    return text
