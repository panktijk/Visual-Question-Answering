''' Author: Aaditya Prakash '''

import json
import pickle
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
import pdb
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# custom
# from utils.get_data import get_test_data, get_train_data
# from utils.arguments import get_arguments
from vqa_model import build_model
from vqa_lstm_model import build_lstm_model


def flatten(image_em_arr):
    return [em for em_list in image_em_arr.tolist() for em in em_list]


def main():

    train_questions = pickle.load(open('vqa_ques_train.pkl', 'rb'))
    questions_df = pd.DataFrame(train_questions)
    # pdb.set_trace()
    images_df = pickle.load(open('vqa_images_train.pkl', 'rb'))
    # images_df = pd.DataFrame(train_images)
    # pdb.set_trace()
    train_data = json.load(open('vqa_train.json', 'r'))
    data_df = pd.DataFrame(train_data)
    
    
    merged_df = pd.merge(data_df, questions_df, how='inner', on=['question_id', 'image_id'])
    merged_df = pd.merge(merged_df, images_df, how='inner', on='image_id')
    # im = np.concatenate(merged_df['image_em'].map(list)).ravel().tolist()
    merged_df['image_em'] = merged_df['image_em'].apply(lambda image_em_arr: flatten(image_em_arr))
    X_train = merged_df['question_em'].map(list) + merged_df['image_em']
    X_train = pd.DataFrame.from_items(zip(X_train.index, X_train.values)).T
    y_train = merged_df['answer']
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    pdb.set_trace()

    # X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.2)
    # clf = LinearSVC()
    # clf.fit(X_train_train, y_train_train)
    # print(clf.score(X_train_test, y_train_test))
    # dataset, test_img_feature,  test_data, val_answers = get_test_data(args)

    # train_X = merged_df[['question_em', 'image_em']]
    # train_Y = merged_df['answer']

    # test_X = [test_data[u'question'], test_img_feature]
    # test_Y = np_utils.to_categorical(val_answers, args.nb_classes)


    # model_name = importlib.import_module("models."+args.model)
    # model = model_name.model(args)
    model = build_lstm_model()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary() # prints model layers with weights

    filepath = "latent_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

    tensorboard = TensorBoard(log_dir='./latent_logs')

    callbacks_list = [checkpoint, tensorboard]

    history = model.fit(X_train, y_train, batch_size=10, nb_epoch=1, callbacks=callbacks_list)

    return history.history

if __name__ == "__main__":
    main()


