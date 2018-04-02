import torch
import json
import pandas as pd
import numpy as np
import h5py
import pickle


def encode_questions(questions):

	infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
	glove_path = 'GloVe/glove.840B.300d.txt'
	infersent.set_glove_path(glove_path)
	infersent.build_vocab(questions, tokenize=True)
	embeddings = infersent.encode(questions, tokenize=True)
	# infersent.visualize(questions[0], tokenize=True)

	return embeddings


def write_to_file(data, embeddings, output_file):

	em_data = []
	for row, em in zip(data, embeddings):
		em_row = {}
		em_row['question_em'] = em
		em_row['question_id'] = row['question_id']
		em_row['image_id'] = row['image_id']
		em_data.append(em_row)

	# f = h5py.File('vqa_ques_train.h5py', 'w')
	# f.create_dataset('ques_train', data=em_data)
	# f.close()
	pickle.dump(em_data, open(output_file, 'wb'))


def get_question_features(input_file, output_file)

	data = json.load(open(input_file, 'r'))
	questions = [data['question'] for data in data]
	embeddings = encode_questions(questions[:10])
	write_to_file(data, embeddings, output_file)

	# with h5py.File('vqa_ques_train.h5py', 'r') as hf:
	# 	temp = np.array(hf.get('ques_train'))
	# temp = pickle.load(open(output_file, 'rb'))

