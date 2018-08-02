import numpy as np;
import matplotlib.pyplot as plt; 
np.random.seed(20);
means = [[-1, -1], [1, -1], [0, 1]]
cov = [[1, 0], [0, 1]]
N = 200;

X0 = np.random.multivariate_normal(means[0], cov, N);
X1 = np.random.multivariate_normal(means[1], cov, N);
X2 = np.random.multivariate_normal(means[2], cov, N);

X = np.concatenate((X0, X1, X2), axis = 0);
y = np.asarray([0]*N + [1]*N + [2]*N);

def softmax_stable(Z):
	e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
	A = e_Z / e_Z.sum(axis = 1, keepdims = True)
	return A

def crossentropy_loss(Yhat, y):
	id0 = range(Yhat.shape[0]);
	return -np.mean(np.log(Yhat[id0, y]));

def mlp_init(d0, d1, d2):
	"""
		we need to initilize W1, W2, b1, b2
		d0 size of input data
		d1 number units of hidden layer
		d2 number units of output layer
	"""

	W1 = 0.01*np.random.rand(d0, d1);
	b1 = np.zeros(d1);
	W2 = 0.02*np.random.rand(d1, d2);
	b2 = np.zeros(d2);
	return (W1, b1, W2, b2);
def mlp_predict(X, W1, b1, W2, b2):
	"""
		suppose that the network has been trained, predict class of new points 
	"""
	Z1 = X.dot(W1) + b1;
	A1 = np.maximum(Z1, 0);
	Z2 = A1.dot(W2) +b2;
	return np.argmax(Z2, axis = 1);

def fit(X, y, W1, b1, W2, b2, eta):
	loss_hist = [];
	for i in range(5000):
		#feedforward
		Z1 = X.dot(W1) + b1;
		A1 = np.maximum(Z1, 0);
		Z2 = A1.dot(W2) + b2;
		Yhat = softmax_stable(Z2);
		if i%1000 == 0:
			loss = crossentropy_loss(Yhat, y);
			print(" iter ", i ," loss ", loss);
			loss_hist.append(loss);
		#back propagaion 
		id0 = range(Yhat.shape[0]);
		Yhat[id0, y] -= 1;
		E2  = Yhat/N;
		dW2 = np.dot(A1.T, E2);
		db2 = np.sum(E2, axis = 0);
		E1  = np.dot(E2, W2.T);
		E1[Z1 <= 0] = 0
		dW1 = np.dot(X.T, E1);
		db1 = np.sum(E1, axis = 0);

		#gradient descent update
		W1  -= eta*dW1;
		b1  -= eta*db1; 		 
		W2  -= eta*dW2;
		b2  -= eta*db2;
	return (W1, b1, W2, b2, loss_hist);

d0 = 2
d1 = h = 100
d2 = C = 3
(W1, b1, W2, b2) = mlp_init(d0, d1, d2);
(W1, b1, W2, b2, loss_hist) = fit(X, y, W1, b1, W2, b2, 0.3);
Y_pred = mlp_predict(X, W1, b1, W2, b2);
acc = 100*np.mean(Y_pred == y);
print("accuracy of model ", acc);




