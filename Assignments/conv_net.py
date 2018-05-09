import numpy as np
import matplotlib.pyplot as plt
import pickle
import hickle as hkl 
import random as rd
import pandas as pd
from scipy.sparse import csr_matrix

np.random.seed(400)

class conv_net:

	"Init the conv net"
	# F_dim = [F_1_dim, F_2_dim]
	# 	F_i_dim = [n_i, k_i]
	#		n_i = number of filters
	#		k_i = number of columns of filter
	#
	# X_dim = (d, n_len) (dimensions of input data)
	# sigmas = [sig_1 (F1), sig_2 (F2), sig_3 (W)]
	def __init__(self, F_dim, X_dim, K, sigmas):
		super(conv_net, self).__init__()

		# hyperparameters
		self.eta = 0.01
		self.rho = 0.95

		# N = 20050 => batch_size = ~100
		self.n_batch = 200

		# How many samples to use from each class
		# 66 is the smallest class
		self.uni_batch_size = 66
		self.uniform = False

		self.train_loss = 0
		self.val_loss = 0

		self.epochs = 30
		self.momentum = True
		self.decay_rate = 0.95

		self.k = [F_dim[0][1], F_dim[1][1]]
		self.n = [F_dim[0][0], F_dim[1][0]]
		self.d = [X_dim[0], self.n[0]]
		self.n_len = X_dim[1]
		self.K = K
		self.opt = True

		# Filters, F_i = [f_{1,1}, ... f_{1,n_i}]
		# F_i_params = [nr_filters, columns, rows]
		self.F_1_params = [self.n[0], self.k[0], self.d[0]]
		# F_1 = (d x k_1 x n_1)
		self.F_1 = self.init_filter(self.F_1_params, sigmas[0])
		self.F_2_params = [self.n[1], self.k[1], self.d[1]]
		# F_2 = (n_1 x k_2 x n_2)
		self.F_2 = self.init_filter(self.F_2_params, sigmas[1])

		self.F = [self.F_1, self.F_2]

		self.n_len_1 = self.n_len - self.k[0] + 1
		self.n_len_2 = self.n_len_1 - self.k[1] + 1

		self.W = self.init_weights(self.K, (self.n[1] * self.n_len_2), sigmas[2])

		# The M_filter matrices for F_1 and F_2
		self.MF = [self.build_M_filter(1), self.build_M_filter(2)]

		#self.MX = self.build_M_input_first(X)
		self.MX = pd.read_pickle('Datasets/M_x_1.pkl')[0]

	# params = (nr_filter, k_i, rows)
	def init_filter(self, params, sigma, Xavier = True):

		n, k, rows = params
		F = np.zeros((n,rows,k))

		if(Xavier):
			F = np.random.normal(0, 2/np.sqrt(rows), (n, rows, k))
		else:
			F = np.random.normal(0, sigma, (n, rows, k))

		return F

	def init_weights(self, K, W_col, sigma, Xavier = True):

		if(Xavier):
			W = np.random.normal(0, 2/np.sqrt(W_col), (K, W_col))
		else:
			W = np.random.normal(0, sigma, (K, W_col))

		return W


	"Generates the batches to use for mini-batch GD"
	# X, Y = the data and labels (one-hot encoded)
	# n_batch = how many batches to use
	# returns:
	#	batches_X,  batches_Y = arrays containging the batches
	def generate_batches(self, X, Y):
		batch_size = int(len(X[0])/self.n_batch)

		batches_X = np.zeros((batch_size, len(X), self.n_batch))
		batches_Y = np.zeros((batch_size, len(Y), self.n_batch))

		for i in range(batch_size):
			start = i*self.n_batch
			end = (i+1)*self.n_batch
			batches_X[i] = X[:,start:end]
			batches_Y[i] = Y[:,start:end]

		return batches_X, batches_Y

	"Generates uniform batches for unbalanced data"
	def generate_uniform_batches(self, X, Y, y):

		classes_X = list()
		classes_Y = list()

		for i in range(self.K):
			classes_X.append(list())
			classes_Y.append(list())

		for i in range(len(X[0])):
			classes_X[y[i]-1].append(X[:,i])
			classes_Y[y[i]-1].append(Y[:,i])

		return classes_X, classes_Y

	"Performs mini-batch gradient decent"
	# GD_params = [eta, n_batch, n_epochs]
	# X = Y = [train, validation]
	# W, b = the initialized weight matrix and bias vector
	# lambda_reg = the panalizing factor for l2-regularization
	# i = which parameter setting (used for the plotting)
	# plot = boolean for plotting
	# returns:
	#	W_star, b_star = the updated weight matrix and bias vector
	def mini_batch_GD(self, X, Y, y):

		n_runs = 100

		train_X, val_X = X
		train_Y, val_Y = Y
		train_y, val_y = y

		if(self.uniform):
			class_X, class_Y = self.generate_uniform_batches(train_X, train_Y, train_y)
			self.train_loss = np.zeros(int(self.epochs/n_runs))
			self.val_loss = np.zeros(int(self.epochs/n_runs))
		else:
			batches_X, batches_Y = self.generate_batches(train_X, train_Y)
			self.train_loss = np.zeros(self.epochs)
			self.val_loss = np.zeros(self.epochs)

		W_star = self.W
		F1_star = np.transpose(self.F[0],(0,2,1)).reshape((1,self.F[0].size))
		F2_star = np.transpose(self.F[1],(0,2,1)).reshape((1,self.F[1].size))

		mom_W = np.zeros((self.K, (self.n[1] * self.n_len_2)))
		mom_F1 = np.zeros(self.F[0].flatten().shape)
		mom_F2 = np.zeros(self.F[1].flatten().shape)

		val_count = 0
		prev_val_loss = 1000
		early_stopped = False
		best_F1 = np.zeros(self.F[0].shape)
		best_F2 = np.zeros(self.F[1].shape)
		best_W = np.zeros(self.W.shape)

		for epoch in range(self.epochs):
			if(self.uniform):
				if(epoch % 100 == 0):
					print("epoch: ", epoch)
			else:
				if(epoch % 1 == 0):
					print("epoch: ", epoch)

			if(self.uniform):
				batch_size = self.uni_batch_size * self.K
				n_batch = 1
				X_batch_c = np.zeros((len(train_X), batch_size))
				Y_batch_c = np.zeros((len(train_Y), batch_size))

				for i in range(self.K):
					idx = np.random.choice(len(class_X[i]), size = self.uni_batch_size, replace=False)
					Xs = np.array(class_X[i])
					Ys = np.array(class_Y[i])

					X_batch_c[:,(i*self.uni_batch_size):((i + 1)*self.uni_batch_size)] = Xs[idx].T
					Y_batch_c[:,(i*self.uni_batch_size):((i + 1)*self.uni_batch_size)] = Ys[idx].T

			else:
				n_batch = self.n_batch

			for batch in range(n_batch):
				if(self.uniform):
					X_batch = X_batch_c
					Y_batch = Y_batch_c
				else:
					X_batch = batches_X[:,:,batch].T
					Y_batch = batches_Y[:,:,batch].T


				S_1, S_2, P = self.forward_pass(X_batch)
				grad_W, grad_F1, grad_F2 = self.back_prop(X_batch, Y_batch, P, S_1, S_2)

				if(self.momentum):
					mom_W = (self.rho * mom_W) + (self.eta * grad_W)
					mom_F1 = (self.rho * mom_F1) + (self.eta * grad_F1)
					mom_F2 = (self.rho * mom_F2) + (self.eta * grad_F2)

					temp_F1 = mom_F1.reshape(self.F_1_params[0], self.F_1_params[1], self.F_1_params[2])
					temp_F2 = mom_F2.reshape(self.F_2_params[0], self.F_2_params[1], self.F_2_params[2])

					self.F[0] -= np.transpose(temp_F1,(0,2,1))
					self.F[1] -= np.transpose(temp_F2,(0,2,1))

					self.W -= mom_W

				else:
					temp_F1 = grad_F1.reshape(self.F_1_params[0], self.F_1_params[1], self.F_1_params[2])
					temp_F2 = grad_F2.reshape(self.F_2_params[0], self.F_2_params[1], self.F_2_params[2])

					grad_F1 = np.transpose(temp_F1,(0,2,1))
					grad_F2 = np.transpose(temp_F2,(0,2,1))

					self.W -= (self.eta * grad_W)
					self.F[0] -= (self.eta * grad_F1)
					self.F[1] -= (self.eta * grad_F2)

			train_loss = self.compute_loss(train_X, train_Y)
			val_loss = self.compute_loss(val_X, val_Y)

			if(self.uniform):
				if(epoch % n_runs == 0):
					epoch_idx = int(epoch/n_runs)
					self.train_loss[epoch_idx] = train_loss
					self.val_loss[epoch_idx] = val_loss

				if(np.absolute(prev_val_loss - val_loss) < 1e-3):
					val_count += 1
				else:
					val_count = 0
					best_F1 = self.F[0]
					best_F2 = self.F[1]
					best_W = self.W
					prev_val_loss = val_loss

				if(val_count > 19):
					early_stopped = True

					f = open("best_F1.bin","wb")
					np.save(f, best_F1)
					f = open("best_F2.bin","wb")
					np.save(f, best_F2)
					f = open("best_W.bin","wb")
					np.save(f, best_W)
					f.close()

					print("Best params found after", epoch, "epochs")
					return 0


			else:
				self.train_loss[epoch] = train_loss
				self.val_loss[epoch] = val_loss

				if(np.absolute(prev_val_loss - val_loss) < 1e-3):
					val_count += 1
				else:
					val_count = 0
					best_F1 = self.F[0]
					best_F2 = self.F[1]
					best_W = self.W
					prev_val_loss = val_loss

				if(val_count > 5):
					early_stopped = True

					f = open("best_F1.bin","wb")
					np.save(f, best_F1)
					f = open("best_F2.bin","wb")
					np.save(f, best_F2)
					f = open("best_W.bin","wb")
					np.save(f, best_W)
					f.close()

					print("Best params found after", epoch, "epochs")
					self.train_loss = self.train_loss[:epoch]
					self.val_loss = self.val_loss[:epoch]

					return 0

			if(self.momentum):
				if(self.uniform):
					if(epoch % 500 == 0):
						self.eta *= self.decay_rate
				else:
					self.eta *= self.decay_rate

		if(not early_stopped):
			print("Completed all training")

			f = open("best_F1.bin","wb")
			np.save(f, best_F1)
			f = open("best_F2.bin","wb")
			np.save(f, best_F2)
			f = open("best_W.bin","wb")
			np.save(f, best_W)
			f.close()

	"Builds the M_filter matrix for gradient computations"
	# filter_nr = specify what filter to compute for (1/2)
	def build_M_filter(self, filter_nr):

		if(filter_nr == 1):
			F = self.F[0]
			num_f, k, d = self.F_1_params 
			n_len = self.n_len
		else:
			F = self.F[1]
			num_f, k, d = self.F_2_params 
			n_len = self.n_len_1

		row_filter = (n_len - k + 1)
		row_M = num_f * row_filter
		col_M = n_len * d
		M_f = np.zeros((row_M, col_M))

		for f_idx, f in enumerate(F):
			v_t = f.flatten('F')
			for i in range(row_filter):
				M_f[(i * num_f) + f_idx, (i*d):(i*d)+len(v_t)] = v_t

		return M_f

	"Builds the M_input matrix for gradient computations"
	# filter_nr = specify what filter nr (1,2)
	# X = vectorized (columnwise, i.e .flatten('F')) input to filter (1/2)	
	def build_M_input(self, filter_nr, X):

		if(filter_nr == 1):
			num_f, k, d = self.F_1_params 
			n_len = self.n_len
		else:
			num_f, k, d = self.F_2_params 
			n_len = self.n_len_1

		rows = n_len - k + 1
		vec = np.zeros((rows, 1, (d*k)))

		M_x = np.zeros(((num_f*rows),(num_f*d*k)))

		for i in range(rows):
			vec[i] = X[(i * d):(k+i)*d]

		for i in range(len(vec)):
			for j in range(num_f):

				row = (i *num_f) + j
				col_start = (j*len(vec[i][0]))
				col_end = (j*len(vec[i][0])) + len(vec[i][0])

				M_x[row, col_start:col_end] = vec[i][0]

		return M_x

	"Pre build the M_input_1 matrices as they wont change"
	# now saved as a pickle-file with sparse matrices
	# that is loaded when the object is initialised
	def build_M_input_first(self, X):

		M_x_sparse = []

		for i in range(len(X[0])):
			mX_i = self.build_M_input(1, X[:,i])
			M_x_sparse.append(csr_matrix(mX_i))

		return M_x_sparse

	"Calculate softmax"
	# returns:
	#	normalized exponential values of vector s
	def softmax(self, s):
		return np.exp(s)/np.sum(np.exp(s), axis=0)

	def forward_pass(self, X):

		M_F_1 = self.build_M_filter(1)
		M_F_2 = self.build_M_filter(2)
		W = self.W

		# X is vectorized (55 * 19) x N => 1045 x N
		# M_F_1 = 300 x 1045
		# S_1 = 300 x N, each column j corresponds
		# to convolving X_j with the filters F_1 = (F_{1,1}, ..., F_{1,n_1}) 
		S_1 = np.maximum(0, (np.matmul(M_F_1, X)))

		# M_F_2 = 260 x 300 
		# S_2 = 260 x N, each column j corresponds
		# to convolving S_1_j with the filters F_2 = (F_{2,1}, ..., F_{2,n_2})
		S_2 = np.maximum(0, (np.matmul(M_F_2, S_1)))

		# W = K x (n_2 * n_len2) = K x 260 
		# s = K x N
		s = np.matmul(W, S_2)

		P = self.softmax(s)

		return S_1, S_2, P

	def compute_loss(self, X_batch, Y_batch):

		B = len(X_batch[0])

		S_1, S_2, P = self.forward_pass(X_batch)
		l_cross = 0
		for data_point in range(B):
			l_cross -= np.log(np.dot(Y_batch[:,data_point], P[:,data_point]))

		J = (l_cross/B)

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

	def compute_class_accuracy(self, X, y):
		S_1, S_2, P = self.forward_pass(X)
		P_star = np.argmax(P, axis=0)
		P_star += 1

		num_per_class = np.zeros(self.K)

		for i in range(len(y)):
			num_per_class[y[i]-1] += 1

		class_acc = np.zeros(self.K)

		for i in range(len(y)):
			if(y[i] == P_star[i]):
				class_acc[y[i]-1] += 1

		class_acc = np.divide(class_acc,num_per_class)

		return class_acc

	def back_prop(self, X, Y, P, S_1, S_2):

		n = len(X[0])
		G = -(Y-P)

		dl_dW = np.zeros((self.K, (self.n[1] * self.n_len_2)))
		dl_dF_1 = np.zeros(self.F_1.flatten().shape)
		dl_dF_2 = np.zeros(self.F_2.flatten().shape)

		# G = K x N, S_2 = 260 x N
		# => dl_dW = K x 260
		dl_dW = (1/n) * np.matmul(G, S_2.T)

		# Propagate grad through fully connected layer
		# and the ReLu in secondd layer
		# G = 260 x N
		G = np.matmul(self.W.T, G)
		H_2 = S_2
		H_2[H_2 > 0] = 1
		G = np.multiply(G, H_2)

		# Compute grad w.r.t second layer conv. filters
		for i in range(n):
			# g_i = 260 x 1
			g_i = G[:, i]
			# x_i = 300 x 1
			x_i = S_1[:, i]
			# M_x_i = 260 x 1200
			M_x_i = self.build_M_input(2, x_i)
			# v = 1 x 1200
			v = np.matmul(g_i.T, M_x_i)

			dl_dF_2 += (1/n)*v

		# Propagate grad to first layer through second layer
		# conv. filters and first layer's Relu

		# M_F_2 = 260 x 300
		# => G = 300 x N
		M_F_2 = self.MF[1]
		G = np.matmul(M_F_2.T, G)

		# H_1 = 300 x N
		H_1 = S_1
		H_1[H_1 > 0] = 1
		G = np.multiply(G, H_1)

		for i in range(n):
			# g_i = 300 x 1
			g_i = G[:, i]
			# x_i = 300 x 1
			x_i = X[:, i]
			# M_x_i = 300 x 5500
			if(self.opt):
				M_x_i = self.MX[i]
				g_i = csr_matrix(g_i.T)
				v = np.array((g_i * M_x_i).toarray())[0]
			else:
				M_x_i = self.build_M_input(1, x_i)
				v = np.matmul(g_i.T, M_x_i)

			dl_dF_1 += (1/n)*v

		return dl_dW, dl_dF_1, dl_dF_2

	def get_confusion_matrix(self, X, y, uniform):

		S_1, S_2, P = self.forward_pass(X)

		P_star = np.argmax(P, axis=0)

		M = np.zeros((self.K, self.K))

		for i in range(len(y)):
			M[y[i] - 1][P_star[i]] += 1

		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.axis('off')
		plt.imshow(M, interpolation='nearest', extent=(1,18,1,18))
		plt.colorbar()

		if(uniform):
			title = "c_mat_uniform_20k-epochs.png"
		else:
			title = "c_mat_regular_20k-epochs.png"	

		plt.savefig(title)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		M = np.diag(np.diag(M))
		ax.axis('off')
		plt.imshow(M, interpolation='nearest', extent=(1,18,1,18))
		plt.colorbar()

		if(uniform):
			title = "c_mat_diag_uniform_20k-epochs.png"
		else:
			title = "c_mat_diag_regular_20k-epochs.png"	

		plt.savefig(title)
		plt.close()


