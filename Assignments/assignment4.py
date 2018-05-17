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

def check_data(X, char_to_ind, n):
	chars = list(char_to_ind.keys())
	string = []
	for i in range(n):
		letter = chars[np.argmax(X[:,i])]
		string.append(letter)

	sequence = ''.join(string)
	print(sequence)


def check_grad():
	RNN, X, char_to_ind = init_rnn(m = 5)
	Y = X[:,1:]

	#for debugging
	X = X[:,:RNN.seq_len]
	Y = Y[:,:RNN.seq_len]

	ngr.check_grad(RNN, X, Y)

def main():
	RNN, X, char_to_ind = init_rnn(m = 100)
	Y = X[:,1:]

	#for debugging
	# X = X[:,:RNN.seq_len]
	# Y = Y[:,:RNN.seq_len]

	h_0 = np.zeros(RNN.m)
	# x_0 = np.zeros(RNN.K)
	# x_0[4] = 1
	# n = 30

	RNN.ada_grad(X, Y, h_0, char_to_ind)
	# seq = RNN.synthesize_seq(h_0, x_0, n, char_to_ind)
	# print(''.join(seq))
	# a, h, o, P = RNN.forward_pass(X, h_0)
	# print(P.shape)
	# print(P)

	# a, h, o, P = RNN.forward_pass(X, h_0)
	# W, U, V = RNN.back_prop(X, Y, a, h, o, P, h_0)

	# seq = RNN.synthesize_seq(h_0, x_0, n, char_to_ind)
	# print(seq)


main()
#check_grad()

