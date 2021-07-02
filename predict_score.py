# import stuff
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from random import randint

import numpy as np
import torch


# Load model
from models import InferSent

from keras.models import load_model

import tensorflow as tf


class model:
	def __init__(self):
		model_version = 2
		MODEL_PATH = "encoder/infersent%s.pkl" % model_version
		params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
		                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
		self.model = InferSent(params_model)
		self.model.load_state_dict(torch.load(MODEL_PATH))
	# Keep it on CPU or put it on GPU
		use_cuda = False
		self.model = self.model.cuda() if use_cuda else self.model
		# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
		W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
		self.model.set_w2v_path(W2V_PATH)
		# Load embeddings of K most frequent words
		self.model.build_vocab_k_words(K=100000)

		self.similarity_model = load_model('trained_model_last_5.h5')

	def similarity_score(self,question1, question2, embeddings_model, similarity_model):
	    
	    embeddings_1 = embeddings_model.encode([question1], bsize=128, tokenize=False, verbose=True)
	    embeddings_2 = embeddings_model.encode([question2], bsize=128, tokenize=False, verbose=True)
	    
	    score = similarity_model.predict([embeddings_1, embeddings_2])
	    
	    return score[0][0]

	def get_score(self,q1,q2):
		# q1 = 'Which fish would survive in salt water?'
		# q2 = 'How can I be a good geologist?'
		# q3 = 'What should I do to be a great geologist?'
		score = self.similarity_score(q1, q2, self.model, self.similarity_model)
		print(q1,q2,score)
		return score