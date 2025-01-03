import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PolynomialRegression:
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
        
    def generate_polynomial_features(self, X, degree):
        #I am writing comments thoroughly for this function, because it was very had to implement the logic into code. Getting to the logic was also no tough, but putting all that to code is tougher
        m, n = X.shape
        names = [chr(97+i) for i in range(n)] #This translates column names to a,b,c,d's
        print(f"The names are {names}")
        comb_list = []
        comb_list.extend(names)
        def gen_comb(this_comb, deg):
            if deg == 1:
                comb_list.extend(this_comb)
                return
            
            new_comb = []
            for term in this_comb:
                for name in names:
                    new_comb.append("".join(sorted(term + name))) #String addition, not number addition
                    
            comb_list.extend(new_comb)
            gen_comb(new_comb, deg-1) #this is recursion, until degree = 1
            
        # for d in range(1,degree+1):
        #     gen_comb(names, 1)
        gen_comb(names, degree)
        comb_list = list(set(comb_list)) #removal of useless duplicate elements
        # print(comb_list)
        X_new = []
        for comb in comb_list:
            indxs = [ord(ch) - 97 for ch in comb] #translating back to indexes. So if comb was 'abc' then it would iterate over each of a,b,c. and return indxs = [97-97, 98-97,99-97] = [0,1,2]
            # since num value of a char is 97
            product = np.prod(X[:, indxs] , axis = 1)
            X_new.append(product)
        X_new = np.column_stack(X_new)

        return X_new
        
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
        J_hist_ep = []
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
            J_hist_ep.append(cost)
            print(f"Epoch: {epoch:03d}, Cost: {cost:.6f}")
        self._plotter(J_hist_ep) if plot_costs else None
        return J_history_batches, J_hist_ep
    
    def k_fold_cv(self, X, Y, k=5,details=True,plot_cost_vs_epoch=True,plot_cost_vs_fold=True,retrain=True,plot_r2_vs_fold=True):
        m = X.shape[0]
        indxs = np.random.permutation(m)
        X_shuffled = X[indxs,:]
        Y_shuffled = Y[indxs]
        Y_shuffled = Y_shuffled.reshape((m,1))
        fold_size = m //k
        train_costs = []
        test_costs = []
        train_r2s = []
        test_r2s = []
        mean_of_train_costs = 0.0
        mean_of_test_costs = 0.0
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
            mean_of_train_costs += (training_cost/k)
            testing_cost = self._compute_cost(x_test,y_test)
            mean_of_test_costs += (testing_cost/k)
            train_r2 = self.R2_score(x_train, y_train)
            test_r2 = self.R2_score(x_test,y_test)
            train_r2s.append(train_r2)
            test_r2s.append(test_r2)
            train_costs.append(training_cost)
            test_costs.append(testing_cost)
            print(f"Training cost for fold {i+1} is {training_cost}")
            print(f"Training R2 score for fold {i+1} is {train_r2}")
            print(f"Testing cost for fold {i+1} is {testing_cost}")
            print(f"Testing R2 score for fold {i+1} is {test_r2}")
            print(f"Fold {i+1} completed")
            print(f"Starting fold {i+2}") if i<k-1 else None
        # mean_of_training_costs = np.mean(training_costs)
        print(f"Mean of Training Costs: {mean_of_train_costs}")
        # mean_of_testing_costs = np.mean(testing_costs)
        print(f"Mean of Testing Costs: {mean_of_test_costs}")
        #Plotting:
        if plot_cost_vs_epoch:
            for i in range(k):
                plt.plot(np.arange(1,len(J_hist_list[i])+1), np.array(J_hist_list[i]),c='r',label='Costs')
                plt.title(f"Cost vs Epoch for fold {i+1}")
                plt.xlabel("Epoch")
                plt.ylabel("Cost")
                plt.legend()
                plt.show()
                
        if plot_cost_vs_fold:
            plt.plot(np.arange(1,k+1), train_costs, c='r', label='Training costs')
            plt.plot(np.arange(1,k+1), test_costs, c='b', label='Testing Costs')
            plt.xlabel('Fold')
            plt.ylabel("Cost")
            plt.legend()
            plt.show()
        if plot_r2_vs_fold:
            plt.plot(np.arange(1,k+1), train_r2s, c='r', label='Training R2 score')
            plt.plot(np.arange(1,k+1), test_r2s, c='b', label='Testing R2 score')
            plt.xlabel('Fold')
            plt.ylabel("R2 score")
            plt.legend()
            plt.show()
        if retrain:
            print("Retraining model...")
            self.train(X, Y, details=details, plot_costs=True)
            print("Retraining Complete")
            
        return train_costs,test_costs, train_r2s, test_r2s
    def _plotter(self, J_hist):
        plt.plot(np.arange(1,len(J_hist)+1), J_hist, c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title("Cost vs Epochs")
        plt.show()
    def predict(self, X):
        return np.dot(X, self.W.T) + self.B
    
    def R2_score(self, X, Y):
        predictions = self.predict(X)
        SS_res = np.sum((Y-predictions)**2)
        Ybar = np.mean(Y)
        SS_tot = np.sum((Y-Ybar)**2)
        R2 = 1-(SS_res/SS_tot)
        return R2
    def residual_plot(self, X, Y):
        # m = np.maximum(X.shape[0],1000)
        predictions = self.predict(X)
        residuals = predictions - Y
        plt.scatter(np.arange(1,X.shape[0]+1), residuals, color='blue', alpha = 0.4)
        plt.axhline(0, color='green', linestyle='--')
        plt.xlabel("Examples")
        plt.ylabel("Residual (y_hat - y)")
        plt.title("Residual plot")
        plt.legend()
        plt.show()
        
    def histogram_plot(self, X, Y):
        # m = np.maximum(X.shape[0],1000)
        predictions = self.predict(X)
        residuals = predictions - Y
        plt.hist(residuals, bins=20,density=True,color='skyblue',edgecolor='black',alpha=0.6)
        plt.xlabel("Residuals (y_hat - y)")
        plt.ylabel("Frequency")
        plt.title("Histogram Plot")
        plt.axvline(0,color='r',linestyle='--')
        plt.legend()
        plt.show()