from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import cv2
import json
import pickle


def encode_images(images):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    model = VGG19(weights='imagenet', include_top=True, pooling='avg')
    embeddings = []
    for img in images:
        img_em = np.zeros((1, 1000))
        # Since VGG was trained as a image of 224x224, every new image
        # is required to go through the same transformation
        im = cv2.resize(cv2.imread(img), (224, 224))
        # im = im.transpose((2,0,1)) # convert the image to RGBA   
        # this axis dimension is required because VGG was trained on a dimension
        # of 1, 3, 224, 224 (first axis is for the batch size
        # even though we are using only one image, we have to keep the dimensions consistent
        im = np.expand_dims(im, axis=0) 
        img_em[0,:] = model.predict(im)[0]
        embeddings.append(img_em)

    return embeddings


def write_to_file(data, embeddings, output_file):

    em_data = []
    for row, em in zip(data, embeddings):
        em_row = {}
        em_row['image_em'] = em
        em_row['image_id'] = row['image_id']
        em_data.append(em_row)

    # f = h5py.File('vqa_ques_train.h5py', 'w')
    # f.create_dataset('ques_train', data=em_data)
    # f.close()
    pickle.dump(em_data, open(output_file, 'wb'))


def get_image_features(input_file, output_file):

    data = json.load(open('vqa_train.json', 'r'))
    images = [data['image_file'] for data in data]
    embeddings = encode_images(images[:5])
    write_to_file(data, embeddings, output_file)

    # with h5py.File('vqa_ques_train.h5py', 'r') as hf:
    #   temp = np.array(hf.get('ques_train'))
    # temp = pickle.load(open(output_file, 'rb'))