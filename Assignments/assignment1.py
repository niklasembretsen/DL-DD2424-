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
	W = np.random.normal(mu, std, (K,d))
	b = np.random.normal(mu, std, K)

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
	W_star = W
	b_star = b

	train_cost = np.zeros(GD_params[2])
	val_cost = np.zeros(GD_params[2])

	for epoch in range(GD_params[2]):
		print("epoch: ", epoch)
		for batch in range(GD_params[1]):
			X_batch = batches_X[batch]
			Y_batch = batches_Y[batch]

			P = forward_pass(X_batch, W_star, b_star)
			grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W_star, lambda_reg)
			W_star = W_star - (GD_params[0] * grad_W)
			b_star = b_star - (GD_params[0] * grad_b)

		if(plot):
			t_cost = compute_cost(X[0], Y[0], W_star, b_star, lambda_reg)
			v_cost = compute_cost(X[1], Y[1], W_star, b_star, lambda_reg)

			# If weights are very small => log(0) in compute cost => inf/NaN
			if(np.isnan(t_cost) or np.isinf(t_cost)):
				t_cost = train_cost[epoch - 1]
			if(np.isnan(v_cost) or np.isinf(v_cost)):
				v_cost = val_cost[epoch - 1]

			train_cost[epoch] = t_cost
			val_cost[epoch] = v_cost

	if(plot):
		plot_cost(train_cost, val_cost, i)

	return W_star, b_star

"Generates the batches to use for mini-batch GD"
# X, Y = the data and labels (one-hot encoded)
# n_batch = how many batches to use
# returns:
#	batches_X,  batches_Y = arrays containging the batches
def generate_batches(X, Y, n_batch):
	batch_size = int(len(X[0])/n_batch)

	batches_X = np.zeros((n_batch, len(X), batch_size))
	batches_Y = np.zeros((n_batch, len(Y), batch_size))

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
def check_grad(X, Y, lambda_reg, h = 1e-6, slow = True, check_size = 10, dim_size = 30720):
	X = X[:,:check_size]
	Y = Y[:,:check_size]

	K = len(Y)
	d = len(X)

	W, b = init_model_params(K,d)

	if(slow):
		num_grad_W, num_grad_b = grad_slow(X, Y, W, b, lambda_reg, h)
	else:
		num_grad_W, num_grad_b = grad(X, Y, W, b, lambda_reg, h)

	P = forward_pass(X, W, b)
	grad_W, grad_b = compute_gradients(X, Y, P, W, lambda_reg)
	epsilon = 1e-10

	comp_W = np.zeros(W.shape)
	comp_b = np.zeros(b.shape)

	for i in range(len(comp_W)):
		for j in range(len(comp_W[0])):
			comp_W[i][j] = abs(grad_W[i][j] - \
				num_grad_W[i][j])/max(epsilon, abs(grad_W[i][j]) \
				 + abs(num_grad_W[i][j]))

	for i in range(len(comp_b)):
		comp_b[i] = abs(num_grad_b[i] - \
			grad_b[i])/max(epsilon, abs(num_grad_b[i]) + abs(grad_b[i]))


	for i in range(len(num_grad_b)):
		print("n: ", num_grad_b[i], "a: ", grad_b[i])

	print("max W:", np.max(comp_W))
	print("max b:", np.max(comp_b))
	print("#wrong W:", np.sum(comp_W > 1e-6))
	print("# wrong b:",np.sum(comp_b > 1e-6))

"Compute numerical gradients (SLOW)"
def grad_slow(X, Y, W, b, lambda_reg, h=1e-6):

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((len(W), 1))

    for i in range(len(b)):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 = compute_cost(X, Y, W, b_try, lambda_reg)

        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, lambda_reg)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(len(W)):
        for j in range(len(W[0])):
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] - h
            c1 = compute_cost(X, Y, W_try, b, lambda_reg)

            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] + h;
            c2 = compute_cost(X, Y, W_try, b, lambda_reg);

            grad_W[i][j] = (c2-c1) / (2*h);

    return grad_W, grad_b

"Compute numerical gradients"
def grad(X, Y, W, b, lambda_reg, h=1e-6):

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((len(W), 1))

    c = compute_cost(X, Y, W, b, lambda_reg)

    for i in range(len(b)):
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, lambda_reg)
        grad_b[i] = (c2-c) / h

    for i in range(len(W)):   
        for j in range(len(W[0])):
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] + h
            c2 = compute_cost(X, Y, W_try, b, lambda_reg)

            grad_W[i][j] = (c2-c) / h

    return grad_W, grad_b

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
	train_X, train_Y, train_y = read_data('data_batch_1')
	val_X, val_Y, val_y = read_data('data_batch_2')
	test_X, test_Y, test_y = read_data('test_batch')

	K = len(train_Y)
	d = len(train_X)
	X = [train_X, val_X]
	Y = [train_Y, val_Y]

	n_epochs = 300
	n_batch = 100
	lambdas = [0, 0, 0.1, 1]
	etas = [0.1, 0.01, 0.01, 0.01]
	lambdas = [0]
	etas = [0.01]

	accuracy = np.zeros(len(etas))

	for i in range(len(etas)):
		lambda_reg = lambdas[i]
		eta = etas[i]
		GD_params = [eta, n_batch, n_epochs]

		W, b = init_model_params(K, d)
		P = forward_pass(train_X, W, b)

		W_star, b_star = mini_batch_GD(X, Y, GD_params, W, b, lambda_reg, i + 1)
		plot_weights(W_star, i + 1)

		acc_test = compute_accuracy(test_X, test_y, W_star, b_star)
		accuracy[i] = acc_test

	print("Accuracy:")
	for i, acc in enumerate(accuracy):
		print("Param setting:", i + 1, "| Test accuracy:", acc)

def grad_check():
	X, Y, train_y = read_data('data_batch_1')
	lambda_reg = 0
	check_grad(X, Y, lambda_reg, check_size = 10, dim_size=100)


#main()
grad_check()


