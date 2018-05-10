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

"Plots the training and validation cost as a function of epochs"
def plot_cost(train_cost, val_cost, uniform):
	#colors = ["green", "red", "yellow", "blue", "black"]
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	epochs = len(train_cost)
	X = np.linspace(0,epochs,epochs)
	#plt.axis([0, epochs, 1, 3])
	plt.plot(X, train_cost, color = "green", label="Training")
	plt.plot(X, val_cost, color = "red", label="Validation")
	plt.legend()

	if(uniform):
		title = "loss_uniform_2.5k-epochs.png"
	else:
		title = "loss_regular_20k-epochs.png"

	plt.savefig(title)
	plt.close()

def best_mod():
	f = open("best_F1_uni.bin","rb")
	F_1 = np.load(f)
	f = open("best_F2_uni.bin","rb")
	F_2 = np.load(f)
	f = open("best_W_uni.bin","rb")
	W = np.load(f)
	f.close()

	conv, X, Y, y, char_to_ind = init_conv()
	X, Y, y = get_validation_split(X, Y, y)

	conv.F[0] = F_1
	conv.F[1] = F_2
	conv.W = W

	acc, pred_class = conv.compute_accuracy(X[1], y[1])
	print("Accuracy:", acc)
	class_acc = conv.compute_class_accuracy(X[1], y[1])

	print("------ CLASS ACCURACY ----------")
	for c, acc in enumerate(class_acc):
		print("class", c + 1, ":", acc*100)

	print("------ CLASS PRECITIONS --------")
	for idx, num in enumerate(pred_class):
		print("Class", idx + 1, ":", num)

	print("---------------------------------")

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

