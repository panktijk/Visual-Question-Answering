from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
import cv2
import spacy
import nltk
import h5py
import pdb


dataDir     ='Dataset'
versionType ='' # this should be '' when using VQA v2.0 dataset
taskType    ='MultipleChoice' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='abstract_v002'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2015'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir      = '%s/Images/scene_img_%s_%s/' %(dataDir, dataType, dataSubType)


def create_combined_dataset():

    train_data = []

    vqa = VQA(annFile, quesFile)
    im_ids = vqa.getImgIds() 

    for im_id in list(set(im_ids)):
        ann_ids = vqa.getQuesIds(imgIds=im_id, ansTypes='yes/no')
        anns = vqa.loadQA(ann_ids)
        for ann in anns:
            # print(ann)
            row = {}
            row['question_id'] = ann['question_id']
            row['image_id'] = ann['image_id']
            ques, multiple_choices, ans = vqa.getQA(ann) 
            row['question'] = ques
            # row['mc_answers'] = multiple_choices
            row['answer'] = ans
            row['image_file'] = imgDir + dataType + '_' + dataSubType + '_'+ str(im_id).zfill(12) + '.png'
            train_data.append(row)

    json.dump(train_data, open('vqa_train.json', 'w'))
    return pd.DataFrame(train_data)


def get_top_answers(vqa_df):

    # g = vqa_df.groupby(['answer'])['question_id'].count().nlargest(10)
    # print(g[0])

    counts = {}
    
    for img in imgs:
        ans = img['answer'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top answer and their counts:')    
    print('\n'.join(map(str,cw[:20])))
    
    vocab = []
    for i in range(2):
        vocab.append(cw[i][1])

    return vocab[:2]


def filter_questions(imgs, ans_to_i):
    new_imgs = []
    for i, img in enumerate(imgs):
        if ans_to_i.get(img['answer'],len(ans_to_i)+1) != len(ans_to_i)+1:
            new_imgs.append(img)

    print('question number reduce from %d to %d '%(len(imgs), len(new_imgs)))
    return new_imgs


def build_vocab_question(imgs):
    # build vocabulary for question and answers.

    count_thr = 0

    for i, img in enumerate(imgs):
        img['processed_tokens'] = nltk.word_tokenize(img['question'].lower())

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    print(imgs)
    print(vocab)
    return imgs, vocab


def encode_question(imgs, wtoi):

    max_length = 20
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['question_id']
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this sequence
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    
    return label_arrays, label_length, question_id


def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['image_file']] = count_img.get(img['image_file'], 0) + 1

    unique_img = [w for w,n in count_img.items()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.


    for i, img in enumerate(imgs):
        img_pos[i] = imgtoi.get(img['image_file'])

    return unique_img, img_pos


# def create_h5py():
#     f = h5py.File('vqa_train.h5py', "w")
#     f.create_dataset("ques_train", dtype='uint32', data=ques_train)
#     f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train)
#     f.create_dataset("answers", dtype='uint32', data=A)
#     f.create_dataset("question_id_train", dtype='uint32', data=question_id_train)
#     f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)


def get_question_features(question):

    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, len(tokens), 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def encode_questions(imgs):
    # ques_vecs = np.zeros((len(imgs), 300))
    ques_vecs = []
    for i, img in enumerate(imgs):
        ques_vecs.append(get_question_features(img['question']))
    return ques_vecs


def get_image_features(image_file_name):

    model = VGG19(weights='imagenet', include_top=True, pooling='avg')
    image_features = np.zeros((1, 1000))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = np.expand_dims(im, axis=0) 
    image_features[0,:] = model.predict(im)[0]
    return image_features


vqa_df = create_combined_dataset()
# imgs = json.load(open('vqa_train.json', 'r'))

# top_ans = get_top_answers(vqa_df)
# ans_to_i = {w:i+1 for i,w in enumerate(top_ans)}
# i_to_ans = {i+1:w for i,w in enumerate(top_ans)}
# imgs_train = filter_questions(imgs, ans_to_i)
# imgs, vocab = build_vocab_question(imgs_train)
# pdb.set_trace()
# unique_img, img_pos = get_unqiue_img(imgs)
# itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
# wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
# ques_train, ques_length_train, question_id_train = encode_question(imgs, wtoi)
# print('='*50)
# print(ques_train)
# print('='*50)
# print(ques_length_train)
# print('='*50)
# print(question_id_train)
# print(encode_questions(imgs_train))
# for im_id in list(set(im_ids))[:2]:
#   print(im_id)
#   ann_ids = vqa.getQuesIds(imgIds=im_id);  
#   # print(annIds)
#   anns = vqa.loadQA(ann_ids)
#   for ann in anns:
#       ques, anss = vqa.getQA(ann)  
#       print(ques)
#       ques_vec = get_question_features(ques)
#       image_file_name = dataType + '_' + dataSubType + '_'+ str(im_id).zfill(12) + '.png'
#       im_vec = get_image_features(image_file_name)
#       print(imgDir + image_file_name)






# with open(quesFile) as f:
#   ques = json.load(f)

# print(len(ques['questions']))

# for q in ques['questions']:
#   print(q)
#   break

# ques_for_im = [q for q in ques['questions'] if q['image_id'] == 0]

# for q in ques_for_im:
#   print(q)