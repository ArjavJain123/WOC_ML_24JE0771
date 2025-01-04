import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:
    def __init__(self,inp_features, alpha=1e-3, reg_param=None, batch_size=32,epochs=100):
        self.epochs = epochs
        self.inp_features = inp_features
        self.alpha = alpha
        self.reg_param = reg_param
        self.W = None
        self.B = None
        self._init_params()
        self.batch_size = batch_size
        self.thresh = 0.5
    def _init_params(self):
        self.W = np.random.randn(1,self.inp_features)
        self.B = 0.0
    def _compute_cost(self, X,Y):
        m = Y.size
        cost = -(1/m)*np.sum(Y*np.log(self.predict_fwb(X)))
        return cost
    def _add_reg_term(self,m):
        reg_cost = np.sum(self.W**2)
        reg_cost = (reg_cost*self.reg_param)/(2*m)
        return reg_cost
    def _get_grads(self,X,Y):
        predictions = self.predict_fwb(X)
        err = predictions - Y
        dj_dw = np.mean(np.multiply(err, X), axis=0)
        dj_db = np.mean(err)
        return dj_dw, dj_db
    def train(self, X, Y, details=False,plot_costs=True,get_report=True,print_con_mat=True):
        m = Y.size
        Y = Y.reshape((m,1))
        J_history_batches = []
        J_hist_ep = []
        for epoch in range(1, self.epochs+1):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation,:]
            Y_shuffled = Y[permutation,:]
            batches = m //self.batch_size
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
                cost = self._compute_cost(mini_batch_X, mini_batch_Y)
                J_history_batches.append(cost)
                cost += self._add_reg_term(mini_batch_Y.size) if self.reg_param is not None else 0
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
                  
                cost = self._compute_cost(mini_batch_X, mini_batch_Y)
                J_history_batches.append(cost)
                cost += self._add_reg_term(mini_batch_Y.size) if self.reg_param is not None else 0
                if details:
                    print(f"Epoch: {epoch:03d}, Batch: last, Cost: {cost:.6f}")
                    
            cost = self._compute_cost(X_shuffled, Y_shuffled)
            J_hist_ep.append(cost)
            cost += self._add_reg_term(m) if self.reg_param is not None else 0
            print(f"Epoch: {epoch:03d}, Cost: {cost:.6f}, Accuracy: {self.get_accuracy(Y_shuffled, self.predict(X_shuffled)):.4f}")
            
        if print_con_mat:
            predictions = self.predict(X)
            con_mat = self.get_conf_mat(Y, predictions)
            self.print_con_mat(con_mat)
        if get_report:
            predictions = self.predict(X)
            self.classification_report(Y, predictions)
        self._plotter(J_hist_ep) if plot_costs else None
        return J_hist_ep
    def k_fold_cv(self, X, Y, k=5, details=False,plot_cost_vs_epoch=True, plot_cost_vs_acc=True,retrain=False):
        m = X.shape[0]
        indxs = np.random.permutation(m)
        X_shuffled = X[indxs,:]
        Y_shuffled = Y[indxs]
        testing_accs = []
        training_accs = []
        if plot_cost_vs_epoch:
            J_hist_list = []
        fold_size = m // k
        for fold in range(1,k+1):
            self._init_params()
            print(f"Working on fold {fold}....")
            test_start = (fold-1)*fold_size
            test_end = fold*fold_size if fold < k-1 else m
            X_train = np.concatenate((X_shuffled[:test_start,:], X_shuffled[test_end:,:]), axis=0)
            Y_train = np.concatenate((Y_shuffled[:test_start], Y_shuffled[test_end:]))
            X_test = X_shuffled[test_start:test_end,:]
            Y_test = Y_shuffled[test_start:test_end]
            X_train_mean = np.mean(X_train, axis=0)
            X_train_std = np.std(X_train, axis=0)
            X_train = (X_train - X_train_mean)/X_train_std
            X_test = (X_test - X_train_mean)/X_train_std
            J_hist_ep = self.train(X_train, Y_train,details=details,plot_costs=False,get_report=False,print_con_mat=False)
            print(f"Training Report for fold {fold}:")
            self.classification_report(Y_train, self.predict(X_train))
            J_hist_list.append(J_hist_ep)
            print(f"Testing report for fold {fold}: ")
            self.classification_report(Y_test, self.predict(X_test))
            test_acc = self.get_accuracy(Y_test, self.predict(X_test))
            train_acc = self.get_accuracy(Y_train, self.predict(X_train))
            testing_accs.append(test_acc)
            training_accs.append(train_acc)
            print(f"Fold {fold} completed!") 
            print(f"Starting fold {fold +1}") if fold<k else None
        mean_test_acc = np.mean(testing_accs)
        mean_train_acc = np.mean(training_accs)
        print(f"Mean of training Accuracies is {mean_train_acc}")
        print(f"Mean of testing accuracies: {mean_test_acc}")    
        if plot_cost_vs_epoch:
            for fold in range(1,k+1):
                plt.plot(np.arange(1,1+len(J_hist_list[fold-1])), J_hist_list[fold-1], c='r',label='Cost')
                plt.xlabel("Epochs")
                plt.ylabel("Costs")
                plt.title(f"Costs vs Epochs for fold {fold}")
                plt.legend()
                plt.show()
                
        if plot_cost_vs_acc:
            plt.plot(np.arange(1,k+1), training_accs, c='blue', label='Training Accuracies', linestyle='--', marker='o')
            plt.plot(np.arange(1,k+1), testing_accs, c='green', label='Testing Accuracies', linestyle='--', marker='o')
            plt.xlabel("Folds")
            plt.ylabel("Accuracies")
            plt.title("Folds vs Accuracies")
            plt.legend()
            plt.show()
            
        return mean_train_acc, mean_test_acc, training_accs, testing_accs
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def predict_fwb(self, X):
        z =  np.dot(X,self.W.T) + self.B
        return self._sigmoid(z)
    def predict(self, X):
        predictions = self.predict_fwb(X)
        pred_final = (predictions >= self.thresh).astype(int)
        return pred_final
    def _plotter(self, J_hist):
        plt.plot(np.arange(1,len(J_hist)+1), J_hist, c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title("Cost vs Epochs")
        plt.show()
        
    def get_conf_mat(self, Y_act, Y_pred):
        n_cls = len(np.unique(Y_act))
        con_mat = np.zeros((n_cls, n_cls))
        for indx, label in enumerate(Y_act):
            con_mat[label, Y_pred[indx]] += 1
        
        return con_mat
    
    def print_con_mat(self, con_mat):
        print("Confusion matrix with predictions on X axis and actual values on Y")
        classes= np.arange(con_mat.shape[0])
        table = pd.DataFrame(con_mat, index=classes, columns=classes)
        print(table)
        
    def get_precision(self,con_mat):
        precs = np.zeros((con_mat.shape[0],))
        for i in range(con_mat.shape[0]):
            precs[i] = con_mat[i,i]/np.sum(con_mat[:,i])
        
        return precs
    
    def get_recall(self, con_mat):
        recs = np.zeros((con_mat.shape[0],))
        for i in range(con_mat.shape[0]):
            recs[i] = con_mat[i,i]/np.sum(con_mat[i,:])
        
        return recs
        
    def get_f1s(self, con_mat):
        precs = self.get_precision(con_mat)
        recs = self.get_recall(con_mat)
        f1s = 2*recs*precs/(recs + precs)
        return f1s
    def get_accuracy(self, Y_act, Y_pred):
        return 100*np.mean(Y_act == Y_pred)
    def classification_report(self, Y_act, Y_pred):
        con_mat = self.get_conf_mat(Y_act, Y_pred)
        precs = self.get_precision(con_mat)
        recs = self.get_recall(con_mat)
        f1s = self.get_f1s(con_mat)
        supports = con_mat.sum(axis=1)
        classes= np.arange(con_mat.shape[0])
        reports = []
        for i in range(con_mat.shape[0]):
            reports.append([precs[i], recs[i], f1s[i], supports[i]])
        table1 = pd.DataFrame(reports,index=classes, columns=['precision','recall','f1-score','support'])
        # print(table1)
        wted_prec = np.average(precs, weights=supports)
        wted_rec = np.average(recs, weights=supports)
        wted_f1 = np.average(f1s, weights=supports)
        tot_support = np.sum(supports)
        wted_data = [[wted_prec,wted_rec,wted_f1,tot_support]]
        table2 = pd.DataFrame(wted_data,index=['Average (weighted)'],columns=['precision', 'recall', 'f1-score', 'support'])
        # print('\n\n', table2)
        table = pd.concat([table1, table2])
        print(table)
        print(f"Accuracy: {self.get_accuracy(Y_pred, Y_act)}%")
