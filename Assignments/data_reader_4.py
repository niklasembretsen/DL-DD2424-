import numpy as np
import pickle
import pandas as pd

def read_data(new = True):

	if(new):
		directory = 'Datasets/'
		file = 'goblet_book.txt'

		path = directory + file

		f = open(path, 'r', encoding='utf-8')

		s = f.read()
		all_chars = list(s)

		C = np.unique(all_chars)
		d = len(C)
		N = len(all_chars)

		# for mapping chars to an index for converting strings to a one-hot encoded matrix
		char_to_ind = dict()
		for i in range(d):
			char_to_ind[C[i]] = i

		X = np.zeros((d, N))

		for i, ch in enumerate(all_chars):
			X[char_to_ind[ch]][i] = 1

		f = open("Datasets/book_chars.bin","wb")
		np.save(f, X)
		f = open("Datasets/char_to_ind.pkl","wb")
		pickle.dump(char_to_ind, f)
		f.close()
	else:
		f = open("Datasets/book_chars.bin","rb")
		X = np.load(f)
		f = open("Datasets/char_to_ind.pkl","rb")
		char_to_ind = pickle.load(f)
		f.close()

	return X, char_to_ind
