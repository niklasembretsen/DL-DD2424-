import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rd
import pandas as pd
import conv_net as conv
import copy

"Compares the analytical and numerical gradients"
# X, Y = the data and labels (one-hot encoded)
# lambda_reg = the panalizing factor for l2-regularization
# h = the small shift used for numerical gradients
# slow = boolean for using the slow numerical gradient
# check_size = number of data points used for computing the gradients
def check_grad(conv, X, Y, h_diff = 1e-5, check_size = 10, dim_size = 1045, to_csv = False):
	X = X[:dim_size,:check_size]
	Y = Y[:dim_size,:check_size]		

	num_F1, num_F2, num_W = num_grad(conv, X, Y, h_diff)

	S_1, S_2, P = conv.forward_pass(X)
	grad_W, grad_F1, grad_F2 = conv.back_prop(X, Y, P, S_1, S_2)

	epsilon = 1e-10

	comp_W = np.zeros(grad_W.shape)
	comp_F1 = np.zeros(grad_F1.shape)
	comp_F2 = np.zeros(grad_F2.shape)

	for i in range(len(comp_W)):
		for j in range(len(comp_W[0])):
			comp_W[i][j] = abs(grad_W[i][j] - \
				num_W[i][j])/max(epsilon, abs(grad_W[i][j]) \
				 + abs(num_W[i][j]))	

	# F_1
	for i in range(len(comp_F1)):
		comp_F1[i] = abs(num_F1[i] - \
			grad_F1[i])/max(epsilon, abs(num_F1[i]) + abs(grad_F1[i]))

	# F_2
	for i in range(len(comp_F2)):
		comp_F2[i] = abs(num_F2[i] - \
			grad_F2[i])/max(epsilon, abs(num_F2[i]) + abs(grad_F2[i]))

	tol_error = 1e-6

	relErr_ratio_W = str(np.sum(comp_W > tol_error)) + '/' + str(comp_W.size)
	max_W = np.max(comp_W)
	relErr_ratio_F1 = str(np.sum(comp_F1 > tol_error)) + '/' + str(comp_F1.size)
	max_F1 = np.max(comp_F1)
	relErr_ratio_F2 = str(np.sum(comp_F2 > tol_error)) + '/' + str(comp_F2.size)
	max_F2 = np.max(comp_F2)

	if(to_csv):
		columns = ['Max relErr', 'relErr > 1e-6']
		rows = ['W', 'F1', 'F2']

		df_relErr = pd.DataFrame([[max_W, relErr_ratio_W], [max_F1, relErr_ratio_F1], [max_F2, relErr_ratio_F2]], columns = columns, index = rows)
		df_relErr.to_csv("relErr.csv")

	print("--------------- RELATIVE ERROR ---------------")
	print("max relError W :", max_W)
	print("max relError F1 :", max_F1)
	print("max relError F2 :", max_F2)

	print("# wrong W :", relErr_ratio_W)
	print("# wrong F1 :", relErr_ratio_F1)
	print("# wrong F2 :", relErr_ratio_F2)

def num_grad(conv, X, Y, h):

	try_conv = copy.deepcopy(conv)
	Gs = [0, 1, 2]

	for l in range(len(conv.F)):
		print("filter: ", l + 1)
		try_conv.F[l] = copy.deepcopy(conv.F[l])

		Gs[l] = (np.zeros(conv.F[l].flatten().shape))
		nf = len(conv.F[l])
	    
		for i in range(nf):
			# n
			try_conv.F[l] = copy.deepcopy(conv.F[l])
			F_try = copy.deepcopy(conv.F[l][i])
			G = np.zeros(F_try.flatten().shape)
	        
			for j in range(len(F_try[0])):
				# d
				for k in range(len(F_try)):
					# k
					F_try1 = copy.deepcopy(F_try)

					F_try1[k][j] = F_try[k][j] - h

					try_conv.F[l][i,k,j] = F_try1[k][j]

					l1 = try_conv.compute_loss(X, Y)
					try_conv.F[l] = copy.deepcopy(conv.F[l])

					F_try2 = copy.deepcopy(F_try)
					F_try2[k][j] = F_try[k][j] + h  

					try_conv.F[l][i,k,j] = F_try2[k][j]

					l2 = try_conv.compute_loss(X, Y) 
					try_conv.F[l] = copy.deepcopy(conv.F[l])  

					G[(j*len(F_try)) + k] = (l2 - l1) / (2*h)

					try_conv.F[l][i, k, j] = F_try[k][j]

			Gs[l][(i * len(G)):((i+1) * len(G))] = G

	# compute the gradient for the fully connected layer
	print("and now 'em weights")
	W_try = copy.deepcopy(conv.W)
	G = np.zeros(W_try.shape);
	for i in range(len(W_try)):
		for j in range(len(W_try[0])):
			W_try1 = copy.deepcopy(W_try)
			W_try1[i][j] = W_try[i][j] - h
			try_conv.W[i][j] = W_try1[i][j]
		            
			l1 = try_conv.compute_loss(X, Y)
			try_conv.W = copy.deepcopy(W_try)
			        
			W_try2 = copy.deepcopy(W_try)
			W_try2[i][j] = W_try[i][j] + h          
			        
			try_conv.W[i][j] = W_try2[i][j];
			l2 = try_conv.compute_loss(X, Y) 
			try_conv.W = copy.deepcopy(conv.W)           
			        
			G[i][j] = (l2 - l1) / (2*h)
			try_conv.W = copy.deepcopy(W_try)

	Gs[2] = G

	return Gs


def grad_check(conv):
	df = pd.read_pickle('Datasets/X_data.pkl')

	X = df[0][0]
	Y = df[0][1]

	check_grad(conv, X, Y, check_size = 10)

f_1_dim = [20, 5]
f_2_dim = [20, 3]

sig_1 = 0.01
sig_2 = 0.01
sig_3 = 0.01

sigmas = [sig_1, sig_2, sig_3]

f_dim = [f_1_dim, f_2_dim]
X_dim = [55, 19]
K = 18

conv = conv.conv_net(f_dim, X_dim, K, sigmas)

grad_check(conv)
