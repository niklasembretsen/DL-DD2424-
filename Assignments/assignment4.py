import numpy as np
import matplotlib.pyplot as plt
import data_reader_4 as dr
import random as rd
import pandas as pd
import num_grad_rnn as ngr
import rnn
import time

def init_rnn(m):

	X, char_to_ind = dr.read_data()
	K = len(X)
	RNN = rnn.RNN(K, m = m)

	return RNN, X, char_to_ind

def check_grad():
	RNN, X, char_to_ind = init_rnn(m = 5)
	Y = X[:,1:]

	#for debugging
	X = X[:,:RNN.seq_len]
	Y = Y[:,:RNN.seq_len]

	ngr.check_grad(RNN, X, Y)

def main():
	RNN, X, char_to_ind = init_rnn()
	Y = X[:,1:]

	#for debugging
	X = X[:,:RNN.seq_len]
	Y = Y[:,:RNN.seq_len]

	a, h, o, P = RNN.forward_pass(X)
	W, U, V = RNN.back_prop(X, Y, a, h, o, P)

	print(np.max(W.flatten()))
	print(np.max(U.flatten()))
	print(np.max(V.flatten()))

	# seq = RNN.synthesize_seq(h_0, x_0, n, char_to_ind)
	# print(seq)


#main()
check_grad()

