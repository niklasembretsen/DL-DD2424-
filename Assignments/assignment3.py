import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rd
import pandas as pd
import conv_net
import time

def init_conv(use_subset = False):
	"Dimensionalities of the filters (number of filters, k_i), d is set from X"
	f_1_dim = [20, 5]
	f_2_dim = [20, 3]
	f_dim = [f_1_dim, f_2_dim]

	"The variance to use when initializing filters/weights if not Xavier"
	sig_1 = 0.01
	sig_2 = 0.01
	sig_3 = 0.01
	sigmas = [sig_1, sig_2, sig_3]

	"Dim of X (d, n_len), and number of classes K"
	X_dim = [55, 19]
	K = 18

	conv = conv_net.conv_net(f_dim, X_dim, K, sigmas)

	"Loading the data set"
	df = pd.read_pickle('Datasets/X_data.pkl')
	char_to_ind = df[0][3]

	if(use_subset):
		subset_size = 1000
		X = np.array(df[0][0][:,:subset_size])
		Y = np.array(df[0][1][:,:subset_size])
		y = np.array(df[0][2][:subset_size])
	else:
		X = np.array(df[0][0])
		Y = np.array(df[0][1])
		y = np.array(df[0][2])

	return conv, X, Y, y, char_to_ind

def get_validation_split(X, Y, y):

	directory = 'Datasets/'
	file = 'Validation_inds_fixed.txt'

	path = directory + file

	f = open(path, 'r')

	s = f.read()
	idx = s.split(' ')

	idx = np.array(idx, dtype=int)

	all_data_idx = np.arange(len(X[0]))
	train_idx = np.delete(all_data_idx, idx)

	train_X = X[:, train_idx]
	val_X = X[:, idx]
	train_Y = Y[:, train_idx]
	val_Y = Y[:, idx]
	train_y = y[train_idx]
	val_y = y[idx]

	return [train_X, val_X], [train_Y, val_Y], [train_y, val_y]

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

"Compare the optimized and non-optimized version"
def comp_main():
	conv, X, Y, y, char_to_ind = init_conv()
	X, Y, y = get_validation_split(X, Y, y)
	conv.uniform = False
	conv.epochs = 1

	opts = [True, False]
	for opt in opts:
		conv.opt = opt
		model = "Optimized" if opt else "Non-optimized"
		start_time = time.time()
		conv.mini_batch_GD(X, Y, y)
		print(model, ": --- %s seconds ---" % (time.time() - start_time))

def pred_names():

	conv, X, Y, y, char_to_ind = init_conv()

	classes = [
		'Arabic', 
		'Chinese',
		'Czech',
		'Dutch',
		'English',
		'French',
		'German',
		'Greek',
		'Irish',
		'Italian',
		'Japanese',
		'Korean',
		'Polish',
		'Portuguese',
		'Russian',
		'Scottish',
		'Spanish',
		'Vietnamese'
	]

	friend_names = ['Embretsen','Lohse', 'Diamant', 'Hrstic', 'Palmborg', 'Rahm']
	names = np.zeros((1045, len(friend_names)))

	for i, n in enumerate(friend_names):
		name = list(n)
		name_i = np.zeros((55, 19))
		for j, char in enumerate(name):
			name_i[char_to_ind[char]][j] = 1

		names[:,i] = name_i.flatten('F')

	model = [True, False]	

	for uni in model:
		if(uni):
			title = 'uni_names.csv'
			f = open("best_F1_uni.bin","rb")
			F_1 = np.load(f)
			f = open("best_F2_uni.bin","rb")
			F_2 = np.load(f)
			f = open("best_W_uni.bin","rb")
			W = np.load(f)
			f.close()
		else:
			title = 'names.csv'
			f = open("best_F1.bin","rb")
			F_1 = np.load(f)
			f = open("best_F2.bin","rb")
			F_2 = np.load(f)
			f = open("best_W.bin","rb")
			W = np.load(f)
			f.close()

		conv.F[0] = F_1
		conv.F[1] = F_2
		conv.W = W

		S_1, S_2, P = conv.forward_pass(names)
		P = P * 100
		P = np.round(P, 4)

		df = pd.DataFrame(P, columns = friend_names, index = classes)
		df.to_csv(title)

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

def main():
	conv, X, Y, y, char_to_ind = init_conv()
	X, Y, y = get_validation_split(X, Y, y)

	epochs = [5000,150]
	uni = [False]
	for i in range(2):
		conv, X, Y, y, char_to_ind = init_conv()
		X, Y, y = get_validation_split(X, Y, y)
		conv.epochs = epochs[i]
		conv.uniform = uni[i]
		conv.mini_batch_GD(X, Y, y)

		model = "Uniform" if uni[i] == True else "Regular"

		print("--------", model, "--------")

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

		conv.get_confusion_matrix(X[1], y[1], uni[i])
		plot_cost(conv.train_loss, conv.val_loss, uni[i])

main()
#comp_main()
#pred_names()
#best_mod()

