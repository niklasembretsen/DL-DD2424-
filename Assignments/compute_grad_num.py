import numpy as np

def grad_slow(X, Y, W, b, lambda_reg, h=1e-6):

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((len(W), 1))

    for i in range(len(b)):
        b_try = b
        b_try[i] = b_try[i] - h
        c1 = compute_cost(X, Y, W, b_try, lambda_reg)
        b_try = b
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, lambda_reg)
        grad_b[i] = (c2-c1) / (h)

    for i in range(len(W)):
        for j in range(len(W[0])):
            W_try = W
            W_try[i][j] = W_try[i][j] - h
            c1 = compute_cost(X, Y, W_try, b, lambda_reg)

            W_try = W
            W_try[i][j] = W_try[i][j] + h;
            c2 = compute_cost(X, Y, W_try, b, lambda_reg);

            grad_W[i][j] = (c2-c1) / (2*h);

    return grad_W, grad_b

def grad(X, Y, W, b, lambda_reg, h):

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((len(W), 1))

    c = compute_cost(X, Y, W, b, lambda_reg)

    for i in range(len(b)):
        b_try = b
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, lambda_reg)
        grad_b[i] = (c2-c) / h

    for i in range(len(W)):   
        for j in range(len(W[0])):
            W_try = W
            W_try[i][j] = W_try[i][j] + h
            c2 = compute_cost(X, Y, W_try, b, lambda_reg)

            grad_W[i][j] = (c2-c) / h

    return grad_W, grad_b
