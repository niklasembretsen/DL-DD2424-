import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rd
import compute_grad_num as num_grad

# Set random seed to get reproducable results
np.random.seed(400)

"Loads the CIFAR-10 data"
# returns:
#	X = image pixel data (d x N)
#	Y = one-hot class representation for each image (K x N)
# 	y = label for each image 0-9 (1 x N)
def read_data(file_name):
	path = 'Datasets/cifar-10/'
	file = path + file_name
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')

	X = dict[b'data']
	#transpose X to get d x N matrix
	X = X.T
	# normalize to get pixel values in [0,1]
	X = X / 255
	y = np.array(dict[b'labels'])

	#get one-hot encoding of labels
	Y = np.zeros((10, len(X[0])))
	for i, label in enumerate(y):
		Y[label][i] = 1 

	return np.array(X), np.array(Y), np.array(y)

"Init model parameters"
# initialize weights and biases to have Gaussian random values
# with mean = mu and standard deviation = std
# returns:
#	W = weight matrix of size K x d
#	b = bias vector of length K
def init_model_params(K, d, mu = 0, std = 1e-2):
	W = np.random.normal(mu, 1/d, (K,d))
	b = np.random.normal(mu, 1/K, K)

	return W, b

"Calculate softmax"
# returns:
#	normalized exponential values of vector s
def softmax(s):
	return np.exp(s)/np.sum(np.exp(s), axis=0)

"Perform a forward pass"
# evaluates the network function,
# i.e softmax(WX + b)
# returns:
#	P - (d x N) matrix of the probabilities for each class
def forward_pass(X, W, b):
	z = np.matmul(W, X)
	b = b.reshape((len(b), 1))
	s = np.add(z, b)
	P = softmax(s)

	return P

"Compute the cost function, J (cross entropy loss)"
# returns:
#	J = sum of cross entropy loss for the network
#	+ a l_2-regulizer 
# returns:
#	The cost of the model, J
def compute_cost(X, Y, W, b, lambda_reg):
	D = len(X[0])
	P = forward_pass(X, W, b)
	l_cross = 0
	for data_point in range(D):
		l_cross -= np.log(np.dot(Y[:,data_point], P[:,data_point]))

	reg_term = (np.square(W)).sum()
	J = (l_cross/D) + (lambda_reg * reg_term)

	return J

"Compute accuracy"
# calc ratio between correct predictions and total
# number of predictions 
# returns:
#	The accuracy of the model, acc (correctly classified/samples)
def compute_accuracy(X, y, W, b):
	p = forward_pass(X, W, b)
	# get columnwise argmax
	p_star = np.argmax(p, axis=0)
	correct = np.sum(p_star == y)
	acc = correct/len(p_star)

	return acc

"Computes the weight and bias gradients"
# returns:
#	The gradients, grad_W, grad_b
def compute_gradients(X, Y, P, W, lambda_reg, batch = True):
	if(batch):
		grad_b = compute_bias_grad_batch(X, Y, P)
		grad_W = compute_weight_grad_batch(X, Y, P, W, lambda_reg)
	else:
		grad_b = compute_bias_grad(X, Y, P)
		grad_W = compute_weight_grad(X, Y, P, W, lambda_reg)

	return grad_W, grad_b

"Computes the bias gradient as a batch"
# returns:
#	The bias gradient, grad_b
def compute_bias_grad_batch(X, Y, P):	

	D = len(X[0])
	# g = [(K x N) - (K x N)].T => (N x K)
	g = -(Y-P).T
	# sum over the n rows => (1 x K)
	g = np.sum(g, axis=0)

	grad_b = (1/D) * g

	return grad_b

"Computes the bias gradient"
# returns:
#	The bias gradient, grad_b
def compute_bias_grad(X, Y, P):
	D = len(X[0])

	grad_b = np.zeros(len(Y))

	for i in range(D):
		#(1 x K) / (1 x K)(K x 1) => 1 x K
		dl_dp = -(Y[:,i]/(np.dot(Y[:,i], P[:,i])))
		# (K x K) - (K x K) => (K x K)
		dp_ds = np.diag(P[:,i]) - np.outer(P[:,i], P[:,i])
		# create identity matrix (K x K)
		ds_dp = np.diag(np.ones(len(Y)))
		# [(1 x K) x (K x K)] x (K x K) => (1 x K)
		dJ_db = np.matmul(np.matmul(dl_dp, dp_ds),ds_dp)

		grad_b = np.add(grad_b, dJ_db)

	return (1/D) * grad_b

"Computes the weight gradient as a batch"
# returns:
#	The weight gradient, grad_W
def compute_weight_grad_batch(X, Y, P, W, lambda_reg):	

	D = len(X[0])
	# g = [(K x N) - (K x N)].T => (N x K)
	g = -(Y-P).T
	# grad_W = [(K x N) x (N x D)] => (K x D)
	grad_W = (1/D) * np.matmul(g.T, X.T)

	return np.add(grad_W, (2*lambda_reg*W))	

"Computes the weight gradient"
# returns:
#	The weight gradient, grad_W
def compute_weight_grad(X, Y, P, W, lambda_reg):
	D = len(X[0])

	grad_W = np.zeros(W.shape)

	for i in range(D):
		# (1 x K) / (1 x K)(K x 1) => 1 x K
		dl_dp = -(Y.T[i]/(np.dot(Y.T[i], P[:,i])))
		# (K x K) - (K x K) => (K x K)
		dp_ds = np.diag(P[:,i]) - np.outer(P[:,i], P[:,i])
		# create identity matrix (K x K)
		ds_dp = np.diag(np.ones(len(P[:,i])))
		# (1 x K) x (K x K) => (1 x K)
		dJ_dz = np.matmul(np.matmul(dl_dp, dp_ds),ds_dp)
		# g : (1 x K)
		g = dJ_dz.reshape(1,len(dJ_dz))
		# x : (d x 1)
		x = X[:,i].reshape(len(X), 1)
		# (K x d) + [(K x 1) x (1 x d)] => (K x d)
		grad_W = np.add(grad_W, (np.matmul(g.T, x.T)))

	grad_W = (1/D) * grad_W

	return np.add(grad_W, (2*lambda_reg*W))

"Performs mini-batch gradient decent"
# GD_params = [eta, n_batch, n_epochs]
# X = Y = [train, validation]
# W, b = the initialized weight matrix and bias vector
# lambda_reg = the panalizing factor for l2-regularization
# i = which parameter setting (used for the plotting)
# plot = boolean for plotting
# returns:
#	W_star, b_star = the updated weight matrix and bias vector
def mini_batch_GD(X, Y, GD_params, W, b, lambda_reg, i, plot = True):
	batches_X, batches_Y = generate_batches(X[0], Y[0], GD_params[1])
	print(batches_X.shape)
	W_star = W
	b_star = b

	train_cost = np.zeros(GD_params[2])
	val_cost = np.zeros(GD_params[2])

	best_W = W
	best_b = b
	min_val = 1000
	check = 0

	for epoch in range(GD_params[2]):
		print("epoch: ", epoch)
		for batch in range(GD_params[1]):
			X_batch = batches_X[:,:,batch].T
			Y_batch = batches_Y[:,:,batch].T
			#print(X_batch.shape)

			P = forward_pass(X_batch, W_star, b_star)
			grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W_star, lambda_reg)
			W_star = W_star - (GD_params[0] * grad_W)
			b_star = b_star - (GD_params[0] * grad_b)

		#t_cost = compute_cost(X[0], Y[0], W_star, b_star, lambda_reg)
		v_cost = compute_cost(X[1], Y[1], W_star, b_star, lambda_reg)

		if(v_cost < min_val):
			print("min: ", min_val, "val: ", v_cost)
			min_val = v_cost
			best_W = W_star
			best_b = b_star
			check = 0
		else:
			print("Worse: ", check)
			check += 1

		if(check > 10):
			return best_W, best_b

		#train_cost[epoch] = t_cost
		val_cost[epoch] = v_cost

		if(plot):
			t_cost = compute_cost(X[0], Y[0], W_star, b_star, lambda_reg)
			v_cost = compute_cost(X[1], Y[1], W_star, b_star, lambda_reg)

			if(v_cost < min_val):
				min_val = v_cost
				best_W = W_star
				best_b = b_star

			train_cost[epoch] = t_cost
			val_cost[epoch] = v_cost

	if(plot):
		plot_cost(train_cost, val_cost, i)

	return best_W, best_b

"Generates the batches to use for mini-batch GD"
# X, Y = the data and labels (one-hot encoded)
# n_batch = how many batches to use
# returns:
#	batches_X,  batches_Y = arrays containging the batches
def generate_batches(X, Y, n_batch):
	batch_size = int(len(X[0])/n_batch)

	batches_X = np.zeros((batch_size, len(X), n_batch))
	batches_Y = np.zeros((batch_size, len(Y), n_batch))

	for i in range(batch_size):
		start = i*n_batch
		end = (i+1)*n_batch
		batches_X[i] = X[:,start:end]
		batches_Y[i] = Y[:,start:end]

	return batches_X, batches_Y

"Plots the training and validation cost as a function of epochs"
def plot_cost(train_cost, val_cost, ind):
	plt.xlabel("Epochs")
	plt.ylabel("Cost")
	epochs = len(train_cost)
	X = np.linspace(1,epochs,epochs)
	plt.plot(X, train_cost, color = "green", label="Training")
	plt.plot(X, val_cost, color = "red", label="Validation")
	plt.legend()
	plt.savefig("cost_plot_" + str(ind) + ".png")
	plt.close()

"Visualizes the final weight representations"
def plot_weights(W, ind):
	plt.figure(figsize=(8.4, 2))
	for i, w in enumerate(W):
		#w = w*255
		plt.subplot(1, 10, i + 1)
		im = w.reshape((3, 32, 32))
		im = im.transpose((1,2,0))
		im = ((im - np.min(im))/(np.max(im) - np.min(im)))
		plt.imshow(im)
		plt.xticks(())
		plt.yticks(())
	plt.suptitle("Visualization of the weight matrix", fontsize=16)
	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
	plt.savefig("weight_plot_" + str(ind) + ".png")
	plt.close()
	print("Weight visualization", ind, "completed")

def main():
	X_1, Y_1, y_1 = read_data('data_batch_1')
	X_2, Y_2, y_2 = read_data('data_batch_2')
	X_3, Y_3, y_3 = read_data('data_batch_3')
	X_4, Y_4, y_4 = read_data('data_batch_4')
	X_5, Y_5, y_5 = read_data('data_batch_5')
	#val_X, val_Y, val_y = read_data('data_batch_2')
	test_X, test_Y, test_y = read_data('test_batch')

	val_for_train = 5000

	K = len(Y_1)
	d = len(X_1)

	X_train = np.array([np.concatenate((X_1, X_2, X_3, X_4, X_5[:,:val_for_train]), axis = 1)]).reshape((3072, (40000 + val_for_train)))
	Y_train = np.array([np.concatenate((Y_1, Y_2, Y_3, Y_4, Y_5[:,:val_for_train]), axis = 1)]).reshape((10, (40000 + val_for_train)))
	X_val = X_5[:,val_for_train:]
	Y_val = Y_5[:,val_for_train:]

	X = [X_train, X_val]
	Y = [Y_train, Y_val]

	n_epochs = 500
	n_batch = 100

	#Optimal values from grid search
	# lambdas = [0.0013]
	# etas = [0.004]
	lambdas = [0.001]
	etas = [0.013 ]

	accuracy = np.zeros((len(etas), len(lambdas)))
	print(n_epochs)
	for i, eta in enumerate(etas):
		for j, lamb in enumerate(lambdas):
			print("Lambda: ", lamb, "Eta: ", eta)
			lambda_reg = lambdas[i]
			eta = etas[i]
			GD_params = [eta, n_batch, n_epochs]

			W, b = init_model_params(K, d)
			P = forward_pass(X[0], W, b)

			W_star, b_star = mini_batch_GD(X, Y, GD_params, W, b, lambda_reg, i + 1, plot=False)
			#plot_weights(W_star, i + 1)

			acc_test = compute_accuracy(test_X, test_y, W_star, b_star)
			accuracy[i][j] = acc_test

	print("Accuracy:")
	for i, acc in enumerate(accuracy):
		for j, a in enumerate(acc):
			print("Eta: ", etas[i], "| Lambda: ", lambdas[j],"| Test accuracy:", a)

main()




