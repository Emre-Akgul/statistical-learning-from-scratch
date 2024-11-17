import numpy as np

class LinearRegression():
    def __init__(self, fit_method='ols', loss_function="rmse", l1=0, l2=0, learning_rate=0.01, epochs=1000, min_step_size=0.001, gradient_descent='batch', batch_size=32):
        """
        Initialize the LinearRegression model with a specified fitting method.

        Parameters:
        - fit_method: The fitting method to use: "ols" for Ordinary Least Squares, "gd" for Gradient Descent.
        - learning_rate: Learning rate for Gradient Descent.
        - loss_function: Loss function to use. rmse for Root Mean Squared Error. Only Root Mean Squared Error is supported for now.
        - l1: L1 regularization parameter.
        - l2: L2 regularization parameter.
        - epochs: Number of epochs for Gradient Descent.
        - min_step_size: Minimum step size for Gradient Descent.
        - gradient_descent: Type of gradient_descent. Possible values: "batch", "stochastic", "mini-batch". 
        - batch_size: Size of batch for mini-bactch gradient descent.

        Notes: 
        - You cant use l1 regularization with ols because there is no closed form solution.
        """

        # general parameters
        self.fit_method = fit_method
        self.loss_function = loss_function

        # regularization parameters
        self.l1 = l1
        self.l2 = l2

        # gradient descent parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_step_size = min_step_size
        self.gradient_descent = gradient_descent
        self.batch_size = batch_size

        # initialize weights to none
        self.weights = None # W0 will be bias.
    
    def calculate_loss(self, y_true, y_pred):
        """
        Calculate the loss function value for the given true and predicted values with respect to loss function type and regularization parameters.

        Parameters:
        - y_true: True target values.
        - y_pred: Predicted target values.
        """

        if self.loss_function == 'rmse':
            loss = np.sqrt(np.mean((y_true - y_pred) ** 2)) + self.l1 * np.sum(np.abs(self.weights)) + self.l2 * np.sum(self.weights ** 2)
        else:
            raise ValueError("loss_function should be either 'mse' or 'mae'")

        return loss
    
    def calculate_gradient(self, X, y):
        """
        Calculate the gradient for the loss function for given X, y_true and y_pred values.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        y_pred = self._predict(X)

        if self.loss_function == 'rmse':
            loss_gradient = - X.T @ (y - y_pred) / (X.shape[0] * np.sqrt(np.mean((y - y_pred) ** 2))) + self.l1 * np.sign(self.weights) + 2 * self.l2 * self.weights
        else:
            raise ValueError("loss_function should be rmse.")

        return loss_gradient

    def fit_ols(self, X, y):
        """
        Fit the model to the data using ordinary least squares fit method by calculating weights by given formula.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        self.weights = np.linalg.inv(X.T @ X + self.l2 * np.identity(X.shape[1])) @ X.T @ y

    def fit_gd(self, X, y):
        if self.gradient_descent == 'batch':
            self.fit_gd_batch(X, y)
        elif self.gradient_descent == 'stochastic':
            self.fit_gd_stochastic(X, y)
        elif self.gradient_descent == 'mini-batch':
            self.fit_gd_mini_batch(X, y)
        else:
            raise ValueError("Incorrect gradient_descent value. Possible values: batch, stochastic, mini-batch.")

    def fit_gd_batch(self, X, y):
        """
        Fit the model to the data using batch gradient descent method by updating weights untill convergence.
        Batch gradients use all the training data for updating weights at each step.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        # Initialize weights
        self.weights = np.random.randn(X.shape[1], ) * 0.01
        self.weights[0] = 0 # Thats what they do in NN
        
        # Gradient Descent Loop
        for _ in range(self.epochs):
            gradient = self.calculate_gradient(X, y)
            self.weights = self.weights - self.learning_rate * gradient

    def fit_gd_stochastic(self, X, y):
        """
        Fit the model to the data using batch gradient descent method by updating weights untill convergence.
        Batch gradients use all the training data for updating weights at each step.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        # Initialize weights
        self.weights = np.random.randn(X.shape[1], ) * 0.01
        self.weights[0] = 0 # Thats what they do in NN
        
        n = X.shape[0]
        current_index = 0
        for epoch in range(self.epochs):
            if epoch % n == 0:
                indices = np.arange(n)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]
            
            current_X, current_y = X[current_index : current_index + 1], y[current_index]
            current_index = (current_index + 1) % n
            gradient = self.calculate_gradient(current_X, current_y)
            self.weights = self.weights - self.learning_rate * gradient

    def fit_gd_mini_batch(self, X, y):
        """
        Fit the model to the data using batch gradient descent method by updating weights untill convergence.
        Batch gradients use all the training data for updating weights at each step.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        # Initialize weights
        self.weights = np.random.randn(X.shape[1], ) * 0.01
        self.weights[0] = 0 # Thats what they do in NN

        n = X.shape[0]
        current_index = 0
        for epoch in range(self.epochs):
            if epoch % n == 0:
                indices = np.arange(n)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]
            
            current_X, current_y = X[current_index : min(current_index + self.batch_size, n)], y[current_index : min(current_index + self.batch_size, n)]
            current_index = min(current_index + self.batch_size, n) % n
            gradient = self.calculate_gradient(current_X, current_y)
            self.weights = self.weights - self.learning_rate * gradient

    def fit(self, X, y):
        """
        Fit the model to the data based on selected fit method.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        # Add bias terms coefficent to the X for easier bias term handling.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        if self.fit_method == 'ols':
            self.fit_ols(X, y)
        elif self.fit_method == 'gd':
            self.fit_gd(X, y)
        else:
            raise ValueError("fit_method should be either 'ols' or 'gd'")


    def predict(self, X):
        """
        Predict the target values for given inputs.

        Parameters:
        - X: Input value array for prediction. Should be numpy array with shape (n_samples, n_features).

        Returns:
        - y: Predictions values for input array X. numpy array with shape (n_samples, )
        """

        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Add bias terms coefficent to the X for prediction.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        y = self._predict(X)
        return y
    
    def _predict(self, X):
        """
        Helper method for gradient descent. Using self.predict add 1s for the biases.
        """
        return X @ self.weights