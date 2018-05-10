import numpy as np
import matplotlib.pyplot as plt
import pickle
import hickle as hkl 
import random as rd
import pandas as pd
from scipy.sparse import csr_matrix

np.random.seed(300)

class RNN:

	"Init the RNN"
	def __init__(self, K, m = 100):
		super(RNN, self).__init__()

		# dimensionality of the hidden state
		self.m = m

		# number of classes (i.e number of chars)
		self.K = K

		#the learning rate
		self.eta = 0.1

		# the length of the sequnces used for training
		self.seq_len = 25

		# bias vectors b and c
		self.b = self.init_bias_vec(m)
		self.c = self.init_bias_vec(K)

		# Weight matrices
		self.W = self.init_weights(m, m)
		self.U = self.init_weights(m, K)
		self.V = self.init_weights(K, m)

		self.grads = [self.W, self.U, self.V, self.b, self.c]

		#gradients
		self.grad_W = np.zeros(self.W.shape)
		self.grad_U = np.zeros(self.U.shape)
		self.grad_V = np.zeros(self.V.shape)
		self.grad_b = np.zeros(self.b.shape)
		self.grad_c = np.zeros(self.c.shape)

		self.grads_1 = [self.grad_W, self.grad_U, self.grad_V, self.grad_b, self.grad_c]

	def init_bias_vec(self, dim):
		return np.zeros(dim)

	def init_weights(self, row, col, sigma = 0.01, Xavier = True):

		if(Xavier):
			W = np.random.normal(0, 2/np.sqrt(col), (row, col))
		else:
			W = np.random.normal(0, sigma, (row, col))

		return W

	"Synthesize a sequence of chars using the network"
	# h_0 = hidden state at time 0
	# x_0 = first (dummy) vector to the RNN
	# n = the length of the generated sequence
	#	returns:
	#		seq = the generated sequence
	def synthesize_seq(self, h_0, x_0, n, char_to_ind):

		seq = []
		chars = list(char_to_ind.keys())
		h_t = h_0
		x_t = x_0

		for i in range(n):
			a_t = np.matmul(self.W, h_t) + np.matmul(self.U, x_0) + self.b
			h_t = np.tanh(a_t)
			o_t = np.matmul(self.V, h_t) + self.c
			p_t = self.softmax(o_t)

			seq.append(np.random.choice(a = chars, p = p_t))

		return seq


	"Calculate softmax"
	# returns:
	#	normalized exponential values of vector s
	def softmax(self, s):
		return np.exp(s)/np.sum(np.exp(s), axis=0)

	def forward_pass(self, X):

		n = len(X[0])
		h_t = np.zeros(self.m)

		a = np.zeros((self.m, n))
		h = np.zeros((self.m, n))
		o = np.zeros((self.K, n))
		P = np.zeros((self.K, n))

		for i in range(n):
			a_t = np.matmul(self.W, h_t) + np.matmul(self.U, X[:,i]) + self.b
			h_t = np.tanh(a_t)
			o_t = np.matmul(self.V, h_t) + self.c
			p_t = self.softmax(o_t)

			a[:,i] = a_t
			h[:,i] = h_t
			o[:,i] = o_t
			P[:,i] = p_t

		return  a, h, o, P

	def compute_loss(self, X, Y):

		B = len(X[0])

		a, h, o, P = self.forward_pass(X)
		l_cross = 0
		for data_point in range(B):
			l_cross -= np.log(np.dot(Y[:,data_point], P[:,data_point]))

		#J = (l_cross/B)
		J = l_cross

		return J

	def compute_accuracy(self, X, y):
		S_1, S_2, P = self.forward_pass(X)
		# get columnwise argmax
		P_star = np.argmax(P, axis=0)
		P_star += 1
		correct = np.sum(P_star == y)
		acc = correct/len(P_star)

		predicted_classes = np.zeros(self.K, dtype=int)

		for i in range(len(X[0])):
			predicted_classes[P_star[i] - 1] += 1

		return acc, predicted_classes

	def back_prop(self, X, Y, a, h, o, P):

		tau = len(X[0])

		dL_dh = np.zeros((tau, self.m))
		dL_da = np.zeros((tau, self.m))

		dL_dW = np.zeros(self.W.shape)
		dL_dU = np.zeros(self.U.shape)
		dL_dV = np.zeros(self.V.shape)
		dL_db = np.zeros(self.b.shape)
		dL_dc = np.zeros(self.c.shape)


		# compute all gradients w.r.t o_t
		# [(C x N) - (C x N)].T => (N x C)
		dL_do = -(Y-P).T

		#sum over all rows in dL_do => (1 x C)
		dL_dc = np.sum(dL_do, axis = 0)

		for t in range(tau):
			# 1 x C
			g_t = dL_do[t].reshape((1, self.K))
			# m x 1
			h_t = h[:,t].reshape((self.m, 1))
			# (C x 1) x (1 x m) => C x m
			dL_dV += np.matmul(g_t.T, h_t.T)

		# (1 x C) x (C x m) => 1 x m
		dL_dh[tau - 1] = np.matmul(dL_do[tau - 1], self.V)

		# (m x m)
		diag_a = np.diag(1 - (np.tanh(a[:,tau - 1])**2))

		# (1 x m) x (m x m) => 1 x m
		dL_da[tau - 1] = np.matmul(dL_dh[tau - 1], diag_a)

		for t in range(tau - 2, -1, -1):
			#(1 x C) x (C x m) + (1 x m) x (m x m) => 1 x m
			dL_dh[t] = np.matmul(dL_do[t], self.V) + np.matmul(dL_da[t + 1], self.W)

			# (m x m)
			diag_a = np.diag(1 - (np.tanh(a[:,t])**2))

			# (1 x m) x (m x m) => 1 x m
			dL_da[t] = np.matmul(dL_dh[t], diag_a)

		h_t = np.zeros((self.m, 1))

		for t in range(tau):

			# 1 x m
			g_t = dL_da[t].reshape((1, self.m))
			# (m x 1) x (1 x m) => m x m
			dL_dW += np.matmul(g_t.T, h_t.T)
			h_t = h[:,t].reshape((self.m, 1))

			# d x 1
			x_t = X[:,t].reshape((self.K, 1))
			# (m x 1) x (1 x d) => m x d
			dL_dU += np.matmul(g_t.T, x_t.T)

			dL_db += dL_da[t]

		# dL_dW = (1/tau) * dL_dW
		# dL_dU = (1/tau) * dL_dU
		# dL_dV = (1/tau) * dL_dV

		return [dL_dW, dL_dU, dL_dV, dL_db, dL_dc]







