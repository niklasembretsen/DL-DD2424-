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
		df.to_pickle('names.pkl')
	else:
		df = pd.read_pickle('Datasets/names.pkl')

	return df

def man_data():
	# df has columns 'names' and 'categories'
	df = read_data(new = False)
	char_list = list(map(list, df['names']))

	all_chars = []
	max_len = 0
	#What is this, lol
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

	


man_data()
