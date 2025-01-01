import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,inp_features, alpha=1e-3, reg_param=None, batch_size=32,epochs=100):
        self.epochs = epochs
        self.inp_features = inp_features
        self.alpha = alpha
        self.reg_param = reg_param
        self.cost_func = self._compute_cost if self.reg_param is None else self._compute_cost_reg
        self.W = None
        self.B = None
        self._init_params()
        self.batch_size = batch_size
    def _init_params(self):
        self.W = np.random.randn(1,self.inp_features)
        self.B = 0.0
    def _compute_cost(self,X,Y):
        m = X.shape[0]
        predictions = self.predict(X)
        cost = np.mean((predictions - Y)**2) / 2
        return cost
    def _compute_cost_reg(self, X, Y):
        m = X.shape[0]
        predictions = self.predict(X)
        cost = np.mean((predictions - Y)**2) / 2
        cost += (self.reg_param*np.sum(self.W**2))/(2*m)
        return cost
    def _get_grads(self,X,Y):
        predictions = self.predict(X)
        err = predictions - Y
        dj_dw = np.mean(err * X, axis = 0)
        dj_dw = dj_dw.reshape(self.W.shape)
        dj_db = np.mean(err)
        return dj_dw, dj_db
    def train(self,X,Y,details=True,plot_costs=True):
        m = X.shape[0]
        Y = Y.reshape((m,1))
        J_history_batches = []
        J_history_entire = []
        for epoch in range(1,self.epochs+1):
            indxs = np.random.permutation(m)
            X_shuffled = X[indxs,:]
            Y_shuffled = Y[indxs,:]
            batches = m // self.batch_size
            for k in range(batches):
                mini_batch_X = X_shuffled[k*self.batch_size:(k+1)*self.batch_size,:]
                mini_batch_Y = Y_shuffled[k*self.batch_size: (k+1)*self.batch_size,:]
                dj_dw, dj_db = self._get_grads(mini_batch_X, mini_batch_Y)
                if self.reg_param is None:
                  self.W -= self.alpha*dj_dw
                  self.B -= self.alpha*dj_db
                else:
                  self.W = self.W*(1-self.alpha*self.reg_param/m)
                  self.W -= self.alpha*dj_dw
                  self.B -= self.alpha*dj_db
                # cost = self._compute_cost(mini_batch_X, mini_batch_Y) if self.reg_param is None else self._compute_cost_reg(mini_batch_X, mini_batch_Y)
                cost = self.cost_func(mini_batch_X, mini_batch_Y)
                J_history_batches.append(cost)
                if details:
                    print(f"Epoch: {epoch:03d}, Batch: {k+1}/{batches}, Cost: {cost:.6f}")

            if m%batches != 0:
                mini_batch_X = X_shuffled[batches*self.batch_size:,:]
                mini_batch_Y = Y_shuffled[batches*self.batch_size]
                dj_dw, dj_db = self._get_grads(mini_batch_X, mini_batch_Y)
                self.W -= self.alpha*dj_dw
                self.B -= self.alpha*dj_db
                if self.reg_param is None:
                  self.W -= self.alpha*dj_dw
                  self.B -= self.alpha*dj_db
                else:
                  self.W = self.W*(1-self.alpha*self.reg_param/m)
                  self.W -= self.alpha*dj_dw
                  self.B -= self.alpha*dj_db
                # cost = self._compute_cost(mini_batch_X, mini_batch_Y) if self.reg_param is None else self._compute_cost_reg(mini_batch_X, mini_batch_Y)
                cost = self.cost_func(mini_batch_X, mini_batch_Y)
                J_history_batches.append(cost)
                if details:
                    print(f"Epoch: {epoch:03d}, Batch: last, Cost: {cost:.6f}")
            # cost = self._compute_cost(X_shuffled, Y_shuffled) if self.reg_param is None else self._compute_cost_reg(X_shuffled, Y_shuffled)
            cost = self.cost_func(X_shuffled, Y_shuffled)
            J_history_entire.append(cost)
            predictions = self.predict(X_shuffled)
            print(f"Epoch: {epoch:03d}, Cost: {cost:.6f}")
        self._plotter(J_history_entire) if plot_costs else None
        return J_history_batches, J_history_entire
    
    def k_fold_cv(self, X, Y, k=5,details=True,plot_cost_vs_epoch=True,plot_cost_vs_fold=True,retrain=True):
        m = X.shape[0]
        indxs = np.random.permutation(m)
        X_shuffled = X[indxs,:]
        Y_shuffled = Y[indxs]
        Y_shuffled = Y_shuffled.reshape((m,1))
        fold_size = m //k
        training_costs = []
        testing_costs = []
        if plot_cost_vs_epoch:
            J_hist_list = []
        for i in range(k):
            print(f"Working on fold {i+1}...:")
            test_start = fold_size*i
            test_end = fold_size*(i+1) if i<k-1 else m
            x_train = np.concatenate((X_shuffled[:test_start,:], X_shuffled[test_end:,:]), axis = 0)
            y_train = np.concatenate((Y_shuffled[:test_start,:], Y_shuffled[test_end:,:]), axis = 0)
            x_test = X_shuffled[test_start:test_end,:]
            y_test = Y_shuffled[test_start:test_end,:]
            self._init_params()
            useless, J_hist = self.train(x_train,y_train,details=details, plot_costs=False)
            if plot_cost_vs_epoch:
                J_hist_list.append(J_hist)
            training_cost = self._compute_cost(x_train, y_train)
            testing_cost = self._compute_cost(x_test,y_test)
            training_costs.append(training_costs)
            testing_costs.append(testing_costs)
            print(f"Training cost for fold {i+1} is {training_cost}")
            print(f"Testing cost for fold {i+1} is {testing_cost}")
            print(f"Fold {i+1} completed")
            print(f"Starting fold {i+2}") if i<k-1 else None
            
        print(f"Mean of Training Costs: {np.mean(training_costs)}")
        print(f"Mean of Testing Costs: {np.mean(testing_costs)}")
        #Plotting:
        if plot_cost_vs_epoch:
            for i in range(k):
                plt.plot(np.arange(1,len(J_hist_list[i])+1), J_hist_list[i],c='r',label='Costs')
                plt.title(f"Cost vs Epoch for fold {i+1}")
                plt.xlabel("Epoch")
                plt.ylabel("Cost")
                plt.legend()
                plt.show()
                
        if plot_cost_vs_fold:
            plt.plot(np.arange(1,k+1), training_costs, c='r', label='Training costs')
            plt.plot(np.arange(1,k+1), testing_cost, c='b', label='Testing Costs')
            plt.xlabel('Fold')
            plt.ylabel("Cost")
            plt.legend()
            plt.show()
            
        if retrain:
            print("Retraining model...")
            self.train(X, Y, details=details, plot_costs=True)
            print("Retraining Complete")
            
        return training_costs,testing_costs
    def _plotter(self, J_hist):
        plt.plot(np.arange(1,len(J_hist)+1), J_hist, c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title("Cost vs Epochs")
        plt.show()
    def predict(self, X):
        return np.dot(X, self.W.T) + self.B