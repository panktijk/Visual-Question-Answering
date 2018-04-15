import torch
import json
import pandas as pd
import numpy as np
import h5py
import pickle
import pdb


def encode_questions(train_questions):

	infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
	glove_path = 'GloVe/glove.840B.300d.txt'
	infersent.set_glove_path(glove_path)
	infersent.build_vocab(train_questions, tokenize=True)
	embeddings = infersent.encode(train_questions, tokenize=True)
	# infersent.visualize(train_questions[0], tokenize=True)

	return embeddings


def write_to_file(train_data, embeddings):

	em_data = []
	for row, em in zip(train_data, embeddings):
		em_row = {}
		em_row['question_em'] = em
		em_row['question_id'] = row['question_id']
		em_row['image_id'] = row['image_id']
		em_data.append(em_row)

	# f = h5py.File('vqa_ques_train.h5py', 'w')
	# f.create_dataset('ques_train', data=em_data)
	# f.close()
	pickle.dump(em_data, open('vqa_ques_train.pkl', 'wb'))


def main():

	train_data = json.load(open('vqa_train.json', 'r'))
	pdb.set_trace()
	train_questions = [data['question'] for data in train_data]
	embeddings = encode_questions(train_questions)
	write_to_file(train_data, embeddings)

	pdb.set_trace()
	# with h5py.File('vqa_ques_train.h5py', 'r') as hf:
	# 	temp = np.array(hf.get('ques_train'))
	temp = pickle.load(open('vqa_ques_train.pkl', 'rb'))


if __name__ == '__main__':
	main()

