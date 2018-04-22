import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rd
import compute_grad_num as num_grad
import pandas as pd

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

	# zero center the data
	x_mean = np.mean(X, axis=1)
	x_mean = x_mean.reshape((len(x_mean), 1))
	X = X - x_mean


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
#	theta = [W_1, b_1, W_2, b_2]
#	W_1 = weight matrices of size m x d
#	W_2 = weight matrices of size K x m
#	b_1 = bias vectors of length m
#	b_2 = bias vectors of length K
def init_model_params(m, d, K, mu = 0, std = 1e-3, xavier = True):

	if(xavier):
		W_1 = np.random.normal(mu, 2/np.sqrt(d), (m,d))
		b_1 = np.random.normal(mu, 2/np.sqrt(m), m)
		W_2 = np.random.normal(mu, 2/np.sqrt(m), (K, m))
		b_2 = np.random.normal(mu, 2/np.sqrt(K), K)
	else:
		W_1 = np.random.normal(mu, std, (m,d))
		b_1 = np.zeros(m)
		W_2 = np.random.normal(mu, std, (K, m))
		b_2 = np.zeros(K)

	theta = [W_1, b_1, W_2, b_2]

	return theta

"Calculate softmax"
# returns:
#	normalized exponential values of vector s
def softmax(s):
	return np.exp(s)/np.sum(np.exp(s), axis=0)

"Perform a forward pass"
# evaluates the network function,
# X = the dataset
# theta = [W_1, b_1, W_2, b_2]
# ----------------------------
# s_1 = W_1X + b_1 (m x N)
# h = max(0, s_1) (ReLu activation function)
# s = W_2h + b2 = ([K x m] x [m x N] => K x N)
# i.e softmax(W_2 x max(0, [WX + b]))
# returns:
#	P - (K x N) matrix of the probabilities for each class
# 	s_1, h for gradient computations
def forward_pass(X, theta):
	b_1 = theta[1].reshape((len(theta[1]), 1))
	b_2 = theta[3].reshape((len(theta[3]), 1))

	s_1 = np.matmul(theta[0], X)
	s_1 = np.add(s_1, b_1)
	h = np.maximum(0, s_1)
	s = np.matmul(theta[2], h)
	s = np.add(s, b_2)
	P = softmax(s)

	return P, h

"Compute the cost function, J (cross entropy loss)"
# returns:
#	J = sum of cross entropy loss for the network
#	+ a l_2-regulizer 
# returns:
#	The cost of the model, J
def compute_cost(X, Y, theta, lambda_reg):
	D = len(X[0])
	P, h = forward_pass(X, theta)
	l_cross = 0
	for data_point in range(D):
		l_cross -= np.log(np.dot(Y[:,data_point], P[:,data_point]))

	reg_term_1 = (np.square(theta[0])).sum()
	reg_term_2 = (np.square(theta[2])).sum()
	reg_term = reg_term_1 + reg_term_2

	J = (l_cross/D) + (lambda_reg * reg_term)

	return J

"Compute accuracy"
# calc ratio between correct predictions and total
# number of predictions 
# returns:
#	The accuracy of the model, acc (correctly classified/samples)
def compute_accuracy(X, y, theta):
	p, h = forward_pass(X, theta)
	# get columnwise argmax
	p_star = np.argmax(p, axis=0)
	correct = np.sum(p_star == y)
	acc = correct/len(p_star)

	return acc

"Computes the weight and bias gradients"
# returns:
#	The gradients, [grad_W_1, grad_W_2], [grad_b_1, grad_b_2]
def compute_gradients(X, Y, P, theta, h, lambda_reg):	

	D = len(X[0])

	grad_W_1 = np.zeros(theta[0].shape)
	grad_b_1 = np.zeros(theta[1].shape)
	grad_W_2 = np.zeros(theta[2].shape)
	grad_b_2 = np.zeros(theta[3].shape)

	for data_point in range(D):
		g = -(Y[:, data_point]-P[:, data_point]).T
		h_i = h[:, data_point]
		h_i = h_i.reshape((1, len(h_i)))
		x_i = X[:, data_point]
		x_i = x_i.reshape((len(x_i), 1))

		grad_b_2 += g

		g = g.reshape((1, len(g)))

		grad_W_2 += np.matmul(g.T, h_i)

		g = np.matmul(g, theta[2])

		h_i[h_i > 0] = 1
		g = np.matmul(g, np.diag(h_i[0]))[0,:]

		grad_b_1 += g
		g = g.reshape((len(g), 1))
		grad_W_1 += np.matmul(g, x_i.T)

	grad_b_1 = (1/D) * grad_b_1
	grad_b_2 = (1/D) * grad_b_2
	grad_W_1 = (1/D) * grad_W_1 + 2*lambda_reg*theta[0]
	grad_W_2 = (1/D) * grad_W_2 + 2*lambda_reg*theta[2]

	return  [grad_W_1, grad_W_2], [grad_b_1, grad_b_2]

"Performs mini-batch gradient decent"
# GD_params = [eta, n_batch, n_epochs]
# X = Y = [train, validation]
# W, b = the initialized weight matrix and bias vector
# lambda_reg = the panalizing factor for l2-regularization
# i = which parameter setting (used for the plotting)
# plot = boolean for plotting
# returns:
#	W_star, b_star = the updated weight matrix and bias vector
def mini_batch_GD(X, Y, GD_params, theta, lambda_reg, i = 1, rho = 0.9, plot = True, momentum = True):
	batches_X, batches_Y = generate_batches(X[0], Y[0], GD_params[1])
	W_star_1 = theta[0]
	b_star_1 = theta[1]
	W_star_2 = theta[2]
	b_star_2 = theta[3]

	decay_rate = 0.95
	eta = GD_params[0]

	mom_W_1 = np.zeros((W_star_1.shape))
	mom_b_1 = np.zeros((b_star_1.shape))
	mom_W_2 = np.zeros((W_star_2.shape))
	mom_b_2 = np.zeros((b_star_2.shape))

	theta_star = [W_star_1, b_star_1, W_star_2, b_star_2]
	if(plot):
		train_cost = np.zeros(GD_params[2] + 1)
		val_cost = np.zeros(GD_params[2] + 1)

	best_theta = theta_star
	min_val = 1000
	check = 0

	for epoch in range(GD_params[2]):
		if(epoch % 10 == 0):
			print("epoch: ", epoch)

		if(plot):
			train_cost[epoch] = compute_cost(X[0], Y[0], theta_star, lambda_reg)
			val_cost[epoch] = compute_cost(X[1], Y[1], theta_star, lambda_reg)

		for batch in range(GD_params[1]):
			X_batch = batches_X[:,:,batch].T
			Y_batch = batches_Y[:,:,batch].T

			P, h = forward_pass(X_batch, theta_star)

			grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, theta_star, h, lambda_reg)

			if(momentum):
				mom_W_1 = (rho * mom_W_1) + (GD_params[0] * grad_W[0])
				mom_b_1 = (rho * mom_b_1) + (GD_params[0] * grad_b[0])
				mom_W_2 = (rho * mom_W_2) + (GD_params[0] * grad_W[1])
				mom_b_2 = (rho * mom_b_2) + (GD_params[0] * grad_b[1])

				W_star_1 = W_star_1 - mom_W_1
				W_star_2 = W_star_2 - mom_W_2
				b_star_1 = b_star_1 - mom_b_1
				b_star_2 = b_star_2 - mom_b_2
			else:
				W_star_1 = W_star_1 - (GD_params[0] * grad_W[0])
				W_star_2 = W_star_2 - (GD_params[0] * grad_W[1])
				b_star_1 = b_star_1 - (GD_params[0] * grad_b[0])
				b_star_2 = b_star_2 - (GD_params[0] * grad_b[1])

			theta_star = [W_star_1, b_star_1, W_star_2, b_star_2]

		v_cost = compute_cost(X[1], Y[1], theta_star, lambda_reg)

		if(v_cost < min_val):
			min_val = v_cost
			best_theta = theta_star
			check = 0
		else:
			check += 1

		if(check == 5):
			eta = eta/2
		elif(check > 10):	
			return best_theta

		if(plot):
			t_cost = compute_cost(X[0], Y[0], theta_star, lambda_reg)
			#v_cost = compute_cost(X[1], Y[1], theta_star, lambda_reg)

			# If weights are very small => log(0) in compute cost => inf/NaN
			if(np.isnan(t_cost) or np.isinf(t_cost)):
				t_cost = train_cost[epoch - 1]
			if(np.isnan(v_cost) or np.isinf(v_cost)):
				v_cost = val_cost[epoch - 1]

			train_cost[epoch + 1] = t_cost
			val_cost[epoch + 1] = v_cost

		if(momentum):
			eta = eta * decay_rate

	if(plot):
		plot_cost(train_cost, val_cost, i, GD_params[0])

	return best_theta

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

"Compares the analytical and numerical gradients"
# X, Y = the data and labels (one-hot encoded)
# lambda_reg = the panalizing factor for l2-regularization
# h = the small shift used for numerical gradients
# slow = boolean for using the slow numerical gradient
# check_size = number of data points used for computing the gradients
def check_grad(X, Y, lambda_reg, h_diff = 1e-5, check_size = 10, dim_size = 3072):
	X = X[:dim_size,:check_size]
	Y = Y[:dim_size,:check_size]

	K = len(Y)
	d = len(X)
	m = 50

	theta = init_model_params(m, d, K)

	num_grad_W, num_grad_b = grad_slow(X, Y, theta, lambda_reg, h_diff)

	P, h = forward_pass(X, theta)
	grad_W, grad_b = compute_gradients(X, Y, P, theta, h, lambda_reg)
	epsilon = 1e-10

	comp_W_1 = np.zeros(theta[0].shape)
	comp_b_1 = np.zeros(theta[1].shape)
	comp_W_2 = np.zeros(theta[2].shape)
	comp_b_2 = np.zeros(theta[3].shape)

	abs_W_1 = np.zeros(theta[0].shape)
	abs_b_1 = np.zeros(theta[1].shape)
	abs_W_2 = np.zeros(theta[2].shape)
	abs_b_2 = np.zeros(theta[3].shape)

	comp_W = [comp_W_1, comp_W_2]
	comp_b = [comp_b_1, comp_b_2]

	abs_W = [abs_W_1, abs_W_2]
	abs_b = [abs_b_1, abs_b_2]

	errors = np.zeros((6,2))

	for layer in range(2):
		for i in range(len(comp_W[layer])):
			for j in range(len(comp_W[layer][0])):
				comp_W[layer][i][j] = abs(grad_W[layer][i][j] - \
					num_grad_W[layer][i][j])/max(epsilon, abs(grad_W[layer][i][j]) \
					 + abs(num_grad_W[layer][i][j]))
				abs_W[layer][i][j] = abs(num_grad_W[layer][i][j]) - abs(grad_W[layer][i][j])	

		for i in range(len(comp_b[layer])):
			comp_b[layer][i] = abs(num_grad_b[layer][i] - \
				grad_b[layer][i])/max(epsilon, abs(num_grad_b[layer][i]) + abs(grad_b[layer][i]))
			abs_b[layer][i] = abs(num_grad_b[layer][i]) - abs(grad_b[layer][i])

		print("layer: ", layer + 1)
		for i in range(len(num_grad_b[layer])):
			print("n: ", num_grad_b[layer][i], "a: ", grad_b[layer][i])

		tol_error = 1e-6 if layer == 0 else 1e-6

		print("---------------", layer + 1, "---------------")
		print("max relError W", layer + 1, ":", np.max(comp_W[layer]))
		print("min relError W", layer + 1, ":", np.min(comp_W[layer]))
		print("max relError b", layer + 1, ":", np.max(comp_b[layer]))
		print("min relError b", layer + 1, ":", np.min(comp_b[layer]))
		print("Tollerated error: ", tol_error)
		print("# wrong W", layer + 1, ":", np.sum(comp_W[layer] > tol_error), "/", comp_W[layer].size)
		print("# wrong b", layer + 1, ":", np.sum(comp_b[layer] > tol_error), "/", comp_b[layer].size)

"Compute numerical gradients (SLOW)"
def grad_slow(X, Y, theta, lambda_reg, h=1e-6):

	grad_W_1 = np.zeros(theta[0].shape)
	grad_b_1 = np.zeros((len(theta[1]), 1))
	grad_W_2 = np.zeros(theta[2].shape)
	grad_b_2 = np.zeros((len(theta[3]), 1))

	for i in range(len(theta[1])):
		b_try = np.copy(theta[1])
		b_try[i] = b_try[i] - h
		c1 = compute_cost(X, Y, [theta[0], b_try, theta[2], theta[3]], lambda_reg)

		b_try = np.copy(theta[1])
		b_try[i] = b_try[i] + h
		c2 = compute_cost(X, Y, [theta[0], b_try, theta[2], theta[3]], lambda_reg)

		grad_b_1[i] = (c2-c1) / (2*h)

	for i in range(len(theta[3])):
		b_try = np.copy(theta[3])
		b_try[i] = b_try[i] - h
		c1 = compute_cost(X, Y, [theta[0], theta[1], theta[2], b_try], lambda_reg)

		b_try = np.copy(theta[3])
		b_try[i] = b_try[i] + h
		c2 = compute_cost(X, Y, [theta[0], theta[1], theta[2], b_try], lambda_reg)

		grad_b_2[i] = (c2-c1) / (2*h)

	for i in range(len(theta[0])):
		for j in range(len(theta[0][0])):
			W_try = np.copy(theta[0])
			W_try[i][j] = W_try[i][j] - h
			c1 = compute_cost(X, Y, [W_try, theta[1], theta[2], theta[3]], lambda_reg)

			W_try = np.copy(theta[0])
			W_try[i][j] = W_try[i][j] + h;
			c2 = compute_cost(X, Y, [W_try, theta[1], theta[2], theta[3]], lambda_reg);

			grad_W_1[i][j] = (c2-c1) / (2*h);

	for i in range(len(theta[2])):
		for j in range(len(theta[2][0])):
			W_try = np.copy(theta[2])
			W_try[i][j] = W_try[i][j] - h
			c1 = compute_cost(X, Y, [theta[0], theta[1], W_try, theta[3]], lambda_reg)

			W_try = np.copy(theta[2])
			W_try[i][j] = W_try[i][j] + h;
			c2 = compute_cost(X, Y, [theta[0], theta[1], W_try, theta[3]], lambda_reg);

			grad_W_2[i][j] = (c2-c1) / (2*h);

	return [grad_W_1, grad_W_2], [grad_b_1, grad_b_2]

"Plots the training and validation cost as a function of epochs"
def plot_cost(train_cost, val_cost, eta, ind = 1):
	colors = ["green", "red", "yellow", "blue", "black"]
	plt.xlabel("Epochs")
	plt.ylabel("Cost")
	epochs = len(train_cost)
	X = np.linspace(0,epochs,epochs)
	#plt.axis([0, epochs, 1, 3])
	plt.plot(X, train_cost, color = "green", label="Training")
	plt.plot(X, val_cost, color = "red", label="Validation")
	plt.legend()
	plt.savefig("cost_plot_ALL_30E" + str(ind) + ".png")
	plt.close()

"Visualizes the final weight representations"
def plot_weights(W, ind):
	plt.figure(figsize=(8.4, 2))
	for i, w in enumerate(W):
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

def grad_check():
	X, Y, train_y = read_data('data_batch_1')
	lambda_reg = 0
	check_grad(X, Y, lambda_reg, check_size = 10, dim_size=100)

def gen_rand_etas(e_min, e_max, samples):
	e = e_min + (e_max - e_min) * np.random.uniform(0,1,samples)
	etas = np.power(10, e)

	return np.sort(etas)

def gen_rand_lambdas(e_min, e_max, samples):
	e = e_min + (e_max - e_min) * np.random.uniform(0,1,samples)
	lambdas = np.power(10, e)

	return np.sort(lambdas)

def main():
	# train_X, train_Y, train_y = read_data('data_batch_1')
	# val_X, val_Y, val_y = read_data('data_batch_2')
	# test_X, test_Y, test_y = read_data('test_batch')

	#Use all batches of data
	X_1, Y_1, y_1 = read_data('data_batch_1')
	X_2, Y_2, y_2 = read_data('data_batch_2')
	X_3, Y_3, y_3 = read_data('data_batch_3')
	X_4, Y_4, y_4 = read_data('data_batch_4')
	X_5, Y_5, y_5 = read_data('data_batch_5')
	#val_X, val_Y, val_y = read_data('data_batch_2')
	test_X, test_Y, test_y = read_data('test_batch')

	val_for_train = 0

	train_X = np.array([np.concatenate((X_1, X_2, X_3, X_4, X_5[:,:val_for_train]), axis = 1)]).reshape((3072, (40000 + val_for_train)))
	train_Y = np.array([np.concatenate((Y_1, Y_2, Y_3, Y_4, Y_5[:,:val_for_train]), axis = 1)]).reshape((10, (40000 + val_for_train)))
	val_X = X_5[:,val_for_train:]
	val_Y = Y_5[:,val_for_train:]

	K = len(train_Y)
	d = len(train_X)
	m = 50

	# theta = [W_1, b_1, W_2, b_2]
	theta = init_model_params(m, d, K, xavier = False)

	# Use only a subset of batch_1
	# subset_size = 100
	# train_X = train_X[:, :subset_size]
	# train_Y = train_Y[:, :subset_size]
	# val_X = val_X[:, :subset_size]
	# val_Y = val_Y[:, :subset_size]

	X = [train_X, val_X]
	Y = [train_Y, val_Y]

	n_epochs = 80
	n_batch = 100

	# lambdas = gen_rand_lambdas(-6, -4, 10)
	# etas = gen_rand_etas(-1.7, -1.15, 10)

	lambdas = [1e-5]
	etas = [0.03]

	accuracy = np.zeros((len(lambdas), len(etas)))

	for lamb in range(len(lambdas)):
		print("lambda: ", lamb + 1, "/", len(lambdas))
		for e in range(len(etas)):
			print("eta: ", e + 1, "/", len(etas))
			lambda_reg = lambdas[lamb]
			eta = etas[e]
			GD_params = [eta, n_batch, n_epochs]

			P, h = forward_pass(train_X, theta)

			theta_star = mini_batch_GD(X, Y, GD_params, theta, lambda_reg)
			v_acc = compute_accuracy(test_X, test_y, theta_star)
			print("acc:", v_acc)
			accuracy[lamb][e] = v_acc

	print("Accuracy:")
	for i, lamb in enumerate(lambdas):
		print("------ lambda = ", lamb, "-----------")
		for j, eta in enumerate(etas):
			print("eta = ", eta, " accuracy = ", accuracy[i][j])

	# df_acc = pd.DataFrame(accuracy, index=lambdas, columns=etas)
	# df_acc.to_csv("grid_search.csv", sep = ',')



main()
#grad_check()


