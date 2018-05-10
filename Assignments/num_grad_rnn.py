import numpy as np
import copy
import pandas as pd


def compute_grads_num(X, Y, RNN, h):

	num_grads = []

	matrix = True
	for param in range(len(RNN.grads)):
		if(param > 2):
			matrix = False

		grad_i = compute_grad_num(X, Y, param, RNN, h, matrix)
		num_grads.append(grad_i)

	return num_grads


def compute_grad_num(X, Y, f, RNN, h, matrix):

	if(matrix):
		grad = np.zeros(RNN.grads[f].shape)
		for i in range(len(grad)):
			for j in range(len(grad[0])):
				RNN_try = copy.deepcopy(RNN)
				RNN_try.grads[f][i][j] = RNN.grads[f][i][j] - h
				l1 = RNN_try.compute_loss(X, Y)
				RNN_try.grads[f][i][j] = RNN.grads[f][i][j] + h
				l2 = RNN_try.compute_loss(X, Y)
				grad[i][j] = (l2-l1)/(2*h);

	else: 
		grad = np.zeros(RNN.grads[f].shape)
		for i in range(len(grad)):
			RNN_try = copy.deepcopy(RNN)
			RNN_try.grads[f][i] = RNN.grads[f][i] - h
			l1 = RNN_try.compute_loss(X, Y)
			RNN_try.grads[f][i] = RNN.grads[f][i] + h
			l2 = RNN_try.compute_loss(X, Y)
			grad[i] = (l2-l1)/(2*h)

	return grad

"Compares the analytical and numerical gradients"
# X, Y = the data and labels (one-hot encoded)
# lambda_reg = the panalizing factor for l2-regularization
# h = the small shift used for numerical gradients
# slow = boolean for using the slow numerical gradient
# check_size = number of data points used for computing the gradients
def check_grad(RNN, X, Y, h_diff = 1e-4, check_size = 25, to_csv = False):
	X = X[:,:check_size]
	Y = Y[:,:check_size]		

	num_grads = compute_grads_num(X, Y, RNN, h_diff)

	a, h, o, P = RNN.forward_pass(X)
	grads = RNN.back_prop(X, Y, a, h, o, P)

	epsilon = 1e-10
	gradient_names = ['W', 'U', 'V', 'b', 'c']

	csv_data = pd.DataFrame(0, index = gradient_names, columns = ['Max relError', 'relErr > 1e-6'], dtype=str)

	for grad in range(len(num_grads)):

		comp = np.zeros(grads[grad].shape)

		if(grad < 3):
			for i in range(len(comp)):
				for j in range(len(comp[0])):
					comp[i][j] = abs(grads[grad][i][j] - \
						num_grads[grad][i][j])/max(epsilon, abs(grads[grad][i][j]) \
						 + abs(num_grads[grad][i][j]))
					#print("a:", grads[grad][i][j], "n:", num_grads[grad][i][j])	

		else:
			for i in range(len(comp)):
				comp[i] = abs(grads[grad][i] - \
					num_grads[grad][i])/max(epsilon, abs(grads[grad][i]) + abs(num_grads[grad][i]))

				#print("a:", grads[grad][i], "n:", num_grads[grad][i])	

		tol_error = 1e-6

		relErr_ratio = str(np.sum(comp > tol_error)) + '/' + str(comp.size)
		max_g = np.max(comp)
		min_g = np.min(comp)

		csv_data.set_value(index = gradient_names[grad], col = 'Max relError', value = max_g)
		csv_data.set_value(index = gradient_names[grad], col = 'relErr > 1e-6', value = relErr_ratio)
		csv_data.to_csv("relErr_rnn.csv")

		print("---------- RELATIVE ERROR for grad:", gradient_names[grad], " ----------")
		print("max relError:", max_g)
		print("min relError:", min_g)
		print("# wrong:", relErr_ratio)

	print(csv_data)

