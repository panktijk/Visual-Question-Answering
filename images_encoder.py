from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import cv2
import json
import pickle
import pdb
import time

model = VGG19(weights='imagenet', include_top=True, pooling='avg')

def encode_image(image):
    img_em = np.zeros((1, 1000))

    im = cv2.resize(cv2.imread(image), (224, 224))

    im = np.expand_dims(im, axis=0) 
    img_em[0,:] = model.predict(im)[0]
    embedding = img_em

    return embedding


def write_to_file(images_df):

    em_data = images_df[['image_id', 'image_em']]
    pdb.set_trace()
    # f = h5py.File('vqa_ques_train.h5py', 'w')
    # f.create_dataset('ques_train', data=em_data)
    # f.close()
    pickle.dump(em_data, open('vqa_images_train.pkl', 'wb'))


def main():

    train_data = json.load(open('vqa_train.json', 'r'))
    train_df = pd.DataFrame(train_data)
    images_df = train_df[['image_file', 'image_id']]
    images_df = images_df.drop_duplicates('image_id')
    images_arr = images_df.as_matrix()
    # images_df = images_df.head()
    encode_image_vec = np.vectorize(encode_image, otypes=[np.ndarray])
    batch_size = 100

    for i in range(0, len(images_arr), batch_size):
        images_arr[i:i+batch_size, 0] = encode_image_vec(images_arr[i:i+batch_size, 0])
        print('Batch ',i,' complete!')
    
    images_df = pd.DataFrame(data=images_arr, columns=['image_em', 'image_id'])
    write_to_file(images_df)

    # # with h5py.File('vqa_ques_train.h5py', 'r') as hf:
    # #   temp = np.array(hf.get('ques_train'))
    temp = pickle.load(open('vqa_images_train.pkl', 'rb'))
    pdb.set_trace()


if __name__ == '__main__':
    main()