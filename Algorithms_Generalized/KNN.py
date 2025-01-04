import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
    
class KNN:
    def __init__(self,distance_metric='euc'):
        if 'euc' in distance_metric or 'l2' in distance_metric:
            self.dist_func = self.euc_dist
        elif 'man' in distance_metric or 'l1' in distance_metric:
            self.dist_func = self.manh_dist
        else:
            raise Exception("Unsupported distance function")
    def euc_dist(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2, axis=1))
    
    def manh_dist(self, x1,x2):
        return np.sum(np.absolute(x1-x2))
    
    def majority_vote(self,neighbrs):
        uniq_labels, cnts = np.unique(neighbrs,return_counts=True)
        majority_label = uniq_labels[np.argmax(cnts)]
        return majority_label
    def predict(self, x_train, y_train,x_test,knn_k=3):
        """
        args:
        X_train --> a 2d array of shape (m,n) where m is no of eg. and n is no of features
        Y_train --> a 1d array of shape (m,)
        X_test --> a 2d array, of shape (m_test,n) where m_test is no of eg in testing
        knn_k --> value of k of KNN
        returns:
        Y_pred --> a 1d array of shape (m_test,)
        """
        Y_pred = []
        for test_pt in x_test:
            distances = self.dist_func(test_pt, x_train)
            sorted_indices = np.argsort(distances)
            k_nearest_negih = y_train[sorted_indices[:knn_k]]
            prediction = self.majority_vote(k_nearest_negih)
            Y_pred.append(prediction)
        Y_pred = np.array(Y_pred)
        return Y_pred
    def k_fold_cv(self, X, Y, k_folds=5, knn_k = 3, details=True,plt_acc=True,print_conf_matrix=True,print_report=True):
        m = Y.shape[0]
        indxs = np.random.permutation(m)
        X_shuffled = X[indxs,:]
        Y_shuffled = Y[indxs]
        fold_size = m // k_folds
        testing_accuracies = []
        training_accuracies = []
        for fold in range(k_folds):
            test_start = fold*fold_size
            test_end = (fold+1)*fold_size if fold < k_folds - 1 else m
            X_train = np.concatenate((X_shuffled[:test_start,:], X_shuffled[test_end:,:]), axis=0)
            Y_train = np.concatenate((Y_shuffled[:test_start], Y_shuffled[test_end:]))
            X_test = X_shuffled[test_start:test_end,:]
            Y_test = Y_shuffled[test_start:test_end]
            training_preds = self.predict(X_train, Y_train, X_train, knn_k)
            testing_preds = self.predict(X_train, Y_train, X_test, knn_k)
            training_acc = self.get_accuracy(Y_train, training_preds)
            training_accuracies.append(training_acc)
            testing_acc = self.get_accuracy(Y_test, testing_preds)
            testing_accuracies.append(testing_acc)
            print(f"Training Accuracy for fold: {fold+1} is {training_acc}") if details else None
            print(f"Testing Accuracy for fold: {fold+1} is {testing_acc}") if details else None
            if print_conf_matrix:
                print(f"Training Confusion Matrix for fold {fold+1}:")
                conf_matx_tr = self.get_conf_mat(Y_train, training_preds)
                self.print_con_mat(conf_matx_tr)
                print(f"Testing Confusion Matrix for fold {fold+1}: ")
                conf_matx_te = self.get_conf_mat(Y_test, testing_preds)
                self.print_con_mat(conf_matx_te)
            if print_report:
                print(f"Training Classification Report for fold {fold+1}:")
                self.classification_report(Y_train, training_preds)
                print(f"Testing Classification Report for fold {fold+1}:")
                self.classification_report(Y_test, testing_preds)
        self._plotter_for_CV(training_accuracies, testing_accuracies) if plt_acc else None
        mean_of_training_accuracies = np.mean(training_accuracies)
        mean_of_testing_accuracies = np.mean(testing_accuracies)
        return training_accuracies, testing_accuracies, mean_of_training_accuracies, mean_of_testing_accuracies
    def _plotter_for_CV(self, training_accuracies, testing_accuracies):
        k_folds = len(training_accuracies)
        plt.plot(np.arange(1,k_folds+1), training_accuracies, c='green',label='Training Accuracy')
        plt.plot(np.arange(1,k_folds+1), testing_accuracies, c='blue',label='Testing Accuracy')
        plt.xlabel("Folds")
        plt.ylabel("Accuracies")
        plt.title("Folds vs Accuracies")
        plt.legend()
        plt.show()
    def get_conf_mat(self, Y_act, Y_pred):
        n_cls = len(np.unique(Y_act))
        con_mat = np.zeros((n_cls, n_cls))
        for indx, label in enumerate(Y_act):
            con_mat[label, Y_pred[indx]] += 1
        
        return con_mat
    
    def get_accuracy(self, Y_act, Y_pred):
        return 100*np.mean(Y_act == Y_pred)
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
        
        
    def get_elbow(self, X, Y,list_of_ks, split_fraction=0.8):
        indxs = np.random.permutation(Y.shape[0])
        X_shuffled = X[indxs]
        Y_shuffled = Y[indxs]
        split_index = int(split_fraction* Y.shape[0])
        # print("split_indx is", split_index)
        X_train, X_test = X_shuffled[:split_index,:], X_shuffled[split_index:,:]
        Y_train, Y_test = Y_shuffled[:split_index], Y_shuffled[split_index:]
        training_accs = []
        testing_accs = []
        for k in list_of_ks:
            Y_pred_train = self.predict(X_train, Y_train, X_train,knn_k=k)
            training_acc = self.get_accuracy(Y_train, Y_pred_train)
            Y_pred_test = self.predict(X_train, Y_train, X_test, knn_k=k)
            testing_acc = self.get_accuracy(Y_test, Y_pred_test)
            training_accs.append(training_acc)
            testing_accs.append(testing_acc)
        plt.plot(list_of_ks, training_accs, linestyle='--',color='green', marker='o',label='Training Accuracies')
        plt.plot(list_of_ks, testing_accs, linestyle='--', color='blue', marker='o', label = 'Testing Accuracies')
        plt.xticks(list_of_ks)
        plt.xlabel('K values')
        plt.ylabel('Accuracies')
        plt.title('Training and Testing Accuracies for various K values')
        plt.legend()
        plt.show()
