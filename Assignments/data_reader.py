import numpy as np
import pickle
import random as rd
import pandas as pd

def read_data(new = False):

	if(new):
		directory = 'Datasets/'
		file = 'ascii_names.txt'

		path = directory + file

		f = open(path, 'r')

		s = f.read()
		names = s.split('\n')


		if(len(names[-1]) < 1):      
			names = names[:-1];

		y = np.zeros(len(names), dtype=int)
		name_list = []
		sep = ' '

		for i in range(len(names)):
			n = names[i].split(sep)
			if(len(n) > 2):
				name_list.append(sep.join(n[:-1]))
			else:
				name_list.append(n[0])

			y[i] = int(n[-1])	

		df = pd.DataFrame(name_list, columns=['names'])
		df['categories'] = y 
		df.to_pickle('Datasets/names.pkl')
	else:
		df = pd.read_pickle('Datasets/names.pkl')

	return df

def man_data():
	# df has columns 'names' and 'categories'
	df = read_data(new = False)
	char_list = list(map(list, df['names']))

	all_chars = []
	max_len = 0

	#What is this, lol. Only gonna run this function once though, phew
	for i in char_list:

		l = len(i)
		if(l > max_len):
			max_len = l

		for j in i:
			all_chars.append(j)

	C = np.unique(all_chars)
	d = len(C)
	n_len = max_len
	K = len(np.unique(df['categories']))
	N = len(df['names'])

	# for mapping chars to an index for converting strings to a one-hot encoded matrix
	char_to_ind = dict()
	for i in range(len(C)):
		char_to_ind[C[i]] = i

	X = np.zeros(((d * n_len), N))

	for i, n in enumerate(df['names']):
		name = list(n)
		x_i = np.zeros((d, n_len))
		for j, char in enumerate(name):
			x_i[char_to_ind[char]][j] = 1

		X[:,i] = x_i.flatten('F')

	y_s = df['categories']
	Y = np.zeros((K, N))
	for i in range(N):
		Y[y_s[i] - 1][i] = 1


	df_data = pd.DataFrame([X, Y, df['categories'], char_to_ind, n_len, K])
	df_data.to_pickle('Datasets/X_data.pkl')



#man_data()
