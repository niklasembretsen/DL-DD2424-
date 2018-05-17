import numpy as np
import matplotlib.pyplot as plt
import pickle
import hickle as hkl 
import random as rd
import pandas as pd
from scipy.sparse import csr_matrix

np.random.seed(400)

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

	def init_bias_vec(self, dim):
		return np.zeros(dim)

	def init_weights(self, row, col, sigma = 0.1, Xavier = False):

		if(Xavier):
			W = np.random.normal(0, 2/np.sqrt((col+row)/2), (row, col))
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
			a_t = np.matmul(self.grads[0], h_t) + np.matmul(self.grads[1], x_0) + self.grads[3]
			h_t = np.tanh(a_t)
			o_t = np.matmul(self.grads[2], h_t) + self.grads[4]
			p_t = self.softmax(o_t)

			cp = np.cumsum(p_t)
			a = np.random.uniform()

			ixs = np.where((cp - a) > 0)
			s = chars[ixs[0][0]]
			#s = np.random.choice(a = chars, p = p_t)
			x_t = np.zeros(len(x_t))
			x_t[char_to_ind[s]] = 1

			seq.append(np.random.choice(a = chars, p = p_t))

		return seq

	"Calculate softmax"
	# returns:
	#	normalized exponential values of vector s
	def softmax(self, s):
		return np.exp(s)/np.sum(np.exp(s), axis=0)

	def forward_pass(self, X, h_0, Y):

		W = self.grads[0]
		U = self.grads[1]
		V = self.grads[2]
		b = self.grads[3]
		c = self.grads[4]

		n = len(X[0])
		h_t = h_0

		a = np.zeros((self.m, n))
		h = np.zeros((self.m, n))
		o = np.zeros((self.K, n))
		P = np.zeros((self.K, n))
		loss = np.zeros(n)

		for i in range(n):
			a_t = np.matmul(W, h_t) + np.matmul(U, X[:,i]) + b
			h_t = np.tanh(a_t)
			o_t = np.matmul(V, h_t) + c
			p_t = self.softmax(o_t)

			a[:,i] = a_t
			h[:,i] = h_t
			o[:,i] = o_t
			P[:,i] = p_t
			loss[i] = np.log(np.dot(Y[:,i], p_t))

		l = -np.sum(loss)

		return  a, h, o, P, l

	def back_prop(self, X, Y, a, h, o, P, h_0):

		tau = len(X[0])

		dL_dh = np.zeros((tau, self.m))
		dL_da = np.zeros((tau, self.m))

		dL_dW = np.zeros(self.W.shape)
		dL_dU = np.zeros(self.U.shape)
		dL_dV = np.zeros(self.V.shape)
		dL_db = np.zeros(self.b.shape)
		dL_dc = np.zeros(self.c.shape)

		W = self.grads[0]
		U = self.grads[1]
		V = self.grads[2]
		b = self.grads[3]
		c = self.grads[4]

		# compute all gradients w.r.t o_t
		# [(C x N) - (C x N)].T => (N x C)
		dL_do = -(Y-P).T

		#sum over all rows in dL_do => (1 x C)
		# dL_dc = np.sum(dL_do, axis = 0)

		for t in range(tau):
			# 1 x C
			g_t = dL_do[t,:].reshape((1, self.K))
			# m x 1
			h_t = h[:,t].reshape((self.m, 1))
			# (C x 1) x (1 x m) => C x m
			dL_dV += np.matmul(g_t.T, h_t.T)
			dL_dc += dL_do[t,:]

		# (1 x C) x (C x m) => 1 x m
		dL_dh[tau - 1,:] = np.matmul(dL_do[tau - 1,:], V)

		# (m x m)
		diag_a = np.diag(1 - (np.tanh(a[:,tau - 1])*np.tanh(a[:,tau - 1])))

		# (1 x m) x (m x m) => 1 x m
		dL_da[tau - 1,:] = np.matmul(dL_dh[tau - 1,:], diag_a)

		for t in range(tau - 2, -1, -1):
			#(1 x C) x (C x m) + (1 x m) x (m x m) => 1 x m
			dL_dh[t,:] = np.matmul(dL_do[t,:], V) + np.matmul(dL_da[t + 1,:], W)

			# (m x m)
			diag_a = np.diag(1 - (np.tanh(a[:,t])*np.tanh(a[:,t])))

			# (1 x m) x (m x m) => 1 x m
			dL_da[t,:] = np.matmul(dL_dh[t,:], diag_a)

		h_t = h_0.reshape((self.m, 1))

		for t in range(tau):

			# 1 x m
			g_t = dL_da[t,:].reshape((1, self.m))
			# (m x 1) x (1 x m) => m x m
			#dL_dW += np.matmul(g_t.T, h_t.T)
			dL_dW += np.outer(g_t, h_t)
			h_t = h[:,t].reshape((self.m, 1))
			#h_t = dL_dh[t,:].reshape((self.m, 1))

			# d x 1
			x_t = X[:,t].reshape((self.K, 1))
			# (m x 1) x (1 x d) => m x d
			#dL_dU += np.matmul(g_t.T, x_t.T)
			dL_dU += np.outer(g_t, x_t)

			dL_db += dL_da[t,:]

		grads = [dL_dW, dL_dU, dL_dV, dL_db, dL_dc]

		#gradient clipping
		# for i in range(len(grads)):
		# 	grad = grads[i]
		# 	grad[grad > 1] = 1
		# 	grad[grad < -1] = -1

		return grads

	def ada_grad(self, X_full, Y_full, h_0, char_to_ind):

		seq_len = self.seq_len
		e = 0
		n_iter = 0
		epsilon = 1e-3
		epoch = 0
		names = ['W', 'U', 'V', 'b', 'c']
		smooth_min = 1000
		loss_min = 1000

		a, h, o, P, loss = self.forward_pass(X_full[:,:seq_len], h_0, Y_full[:,:seq_len])
		smooth_loss = loss
		print("Init loss:", smooth_loss)

		m_W = np.zeros(self.W.shape)
		m_U = np.zeros(self.U.shape)
		m_V = np.zeros(self.V.shape)
		m_b = np.zeros(self.b.shape)
		m_c = np.zeros(self.c.shape)

		m = [m_W, m_U, m_V, m_b, m_c]

		while(epoch < 10):
			X = X_full[:,e:e + seq_len]
			Y = Y_full[:,e + 1:e + seq_len + 1]

			if(e == 0):
				h_prev = h_0

			a, h, o, P, loss = self.forward_pass(X, h_prev, Y)
			gradients = self.back_prop(X, Y, a, h, o, P, h_prev)

			for i in range(len(gradients)):
				m[i] = m[i] + np.square(gradients[i])
				self.grads[i] = self.grads[i] - ((self.eta/(np.sqrt(m[i] + epsilon))) * gradients[i])

			# for i in range(len(self.grads)):
			# 	self.grads[i][self.grads[i] > 1] = 1
			# 	self.grads[i][self.grads[i] < -1] = -1

			n_iter += 1
			smooth_loss = (0.999 * smooth_loss) + (0.001 * loss)

			if(smooth_loss < smooth_min):
				smooth_min = smooth_loss

			if(loss < loss_min):
				loss_min = loss

			if(n_iter % 100 == 0):
				print("Epoch:", epoch + 1,"update step:", n_iter, "smooth loss:", smooth_loss)

			if(n_iter % 500 == 0):
				seq = self.synthesize_seq(h_prev, X[:,0], 200, char_to_ind)
				sequence = ''.join(seq)
				print(sequence)

			h_prev = h[:,-1]
			e = e + seq_len
			if(e > (len(X_full[0]) - seq_len)):
				print("-------- DONE WITH EPOCH --------")
				e = 0
				epoch += 1

		print("Lowest loss", loss_min)
		print("lowest smooth", smooth_min)





