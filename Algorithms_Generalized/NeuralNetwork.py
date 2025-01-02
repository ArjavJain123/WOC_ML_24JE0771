import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

   
class NeuralNetwork:
  def __init__(self, layer_list, act_funcs, alpha = 3e-2, batch_size=32, epochs=100, reg_param=None):
    self.layer_list = layer_list
    self.alpha = alpha
    self.batch_size = batch_size
    self.epochs = epochs
    self.params = {}
    self.cache = {}
    self.reg_param = reg_param
    self.grads = {}
    self.act_funcs = act_funcs
    self._init_params(log_it=True)

  #Useful funcs:
  def _relu(self, z):
    return np.maximum(z,0)
  def _softmax(self, z):
    A = np.exp(z)/sum(np.exp(z))
    return A
  def _sigmoid(self, z):
    return 1/(1+np.exp(-z))
  def _reluDer(self, z):
    return (z >0).astype(int)
  def _get_deltaZ_lastLayer(self,A, Y):
    return A - Y
  def _compute_cost(self,A, Y, cost_func):
    if 'soft' in cost_func:
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost
    elif 'sigm' in cost_func:
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))
        return cost
    else:
        raise Exception("Unsupported Cost function")
      
  def _compute_cost_reg(self,A,Y,cost_func):
    m = Y.shape[1]
    cost = self._compute_cost(A,Y,cost_func)
    L = len(self.layer_list) - 1
    reg_cost = 0.0
    for l in range(1,L+1):
      reg_cost += np.sum(self.params[f"W{l}"]**2)
    reg_cost = (reg_cost*self.reg_param)/(2*m)
    cost += reg_cost
    return cost
  def _one_hot(self, Y):
    uniq_labels = np.unique(Y)
    C = len(uniq_labels)
    m = Y.size
    one_hot_Y = np.zeros((C, m))
    label_to_index = {label: index for index, label in enumerate(uniq_labels)}
    indices = [label_to_index[label] for label in Y]
    one_hot_Y[indices, np.arange(m)] = 1
    # for i in range(m):
    #   one_hot_Y[Y[i], i] = 1
    return one_hot_Y

    #params create:
  def _init_params(self,log_it=True):
    L = len(self.layer_list)
    for l in range(1,L):
      self.params[f"W{l}"] = np.random.randn(self.layer_list[l], self.layer_list[l-1]) * np.sqrt(2 / self.layer_list[l-1]) #He method
      self.params[f"b{l}"] = np.random.randn(self.layer_list[l], 1) * np.sqrt(2 / self.layer_list[l-1])
    print("Parameters initialized: ", {key: val.shape for key, val in self.params.items()}) if log_it else None
  def _fwd_prp(self, X):
    self.cache["A0"] = X
    L = len(self.layer_list) - 1
    for l in range(1, L+1):
      Z = np.dot(self.params[f"W{l}"], self.cache[f"A{l-1}"]) + self.params[f"b{l}"]
      self.cache[f"Z{l}"] = Z
      if "relu" in self.act_funcs[l-1]:
        self.cache[f"A{l}"] = self._relu(Z)
      elif 'sig' in self.act_funcs[l-1]:
        self.cache[f"A{l}"] = self._sigmoid(Z)
      elif 'soft' in self.act_funcs[l-1]:
        self.cache[f"A{l}"] = self._softmax(Z)
      else:
        raise Exception("Unsupported activation function")
    return self.cache[f"A{L}"]
  def _update_grads(self, X, Y):
    #No of layers
    L = len(self.layer_list) - 1
    m = Y.shape[1]
    A_last = self.cache[f"A{L}"]
    if self.act_funcs[L-1] == 'sigmoid' or "softmax":
      dZ = self._get_deltaZ_lastLayer(A_last, Y)
    self.grads[f"dZ{L}"] = dZ
    self.grads[f"dW{L}"] = np.dot(dZ, self.cache[f"A{L-1}"].T) / m
    self.grads[f"db{L}"] = np.sum(dZ, axis = 1, keepdims = True)
    if self.reg_param is not None:
      self.grads[f"dW{L}"] += (self.reg_param*self.params[f"W{L}"])/(2*m)
    for l in reversed(range(1,L)):
      dA = np.dot(self.params[f"W{l+1}"].T, dZ) #this is the dA of the current layer
      Z = self.cache[f"Z{l}"]
      A = self.cache[f"A{l}"]
      activation = self.act_funcs[l-1] #when l = 1, it is the first hidden layer, correspondingly , activations[0] gives the act func of that layer
      if activation == "relu":
        dZ = dA * self._reluDer(Z)
      elif activation == "sigmoid":
        dZ = dA * (A * (1 - A))
      else:
        raise ValueError(f"Unsupported activation function: {activation}")

      self.grads[f"dZ{l}"] = dZ
      self.grads[f"dW{l}"] = (1/m)*np.dot(dZ, self.cache[f"A{l-1}"].T)
      self.grads[f"db{l}"] = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
      if self.reg_param is not None:
        self.grads[f"dW{l}"] += (self.reg_param*self.params[f"W{l}"])/(2*m)

    # return self.grads
  def _update_params(self):
      L = len(self.layer_list) - 1
      for l in range(1, L + 1):
          self.params[f"W{l}"] -= self.alpha*self.grads[f"dW{l}"]
          self.params[f"b{l}"] -= self.alpha*self.grads[f"db{l}"]


  def train(self, X, Y, cost_func = 'soft',details=True,plot_costs=True):
    m = X.shape[1]
    if 'sigm' in self.act_funcs[-1]:
      Y = Y.reshape(1,m)
    else:
      Y = self._one_hot(Y)
    J_history_batches = []
    J_history_entire = []
    for epoch in range(1, self.epochs+1):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        batches = m // self.batch_size
        for k in range(0, batches):
            mini_batch_X = X_shuffled[:, k*self.batch_size:(k+1)*self.batch_size]
            mini_batch_Y = Y_shuffled[:, k*self.batch_size:(k+1)*self.batch_size]

            A_last = self._fwd_prp(mini_batch_X)
            self._update_grads(mini_batch_X, mini_batch_Y)
            self._update_params()
            cost = self._compute_cost(A_last, mini_batch_Y, cost_func)
            cost_reg = self._compute_cost_reg(A_last, mini_batch_Y,cost_func) if self.reg_param is not None else cost
            J_history_batches.append(cost)
            if details:
                print(f"Epoch: {epoch:03d}, Batch: {k+1}/{batches}, Cost: {cost_reg:.6f}")
        if m % batches != 0:
            mini_batch_X = X_shuffled[:, batches*self.batch_size:m]
            mini_batch_Y = Y_shuffled[:, batches*self.batch_size:m]

            A_last = self._fwd_prp(mini_batch_X)
            self._update_grads(mini_batch_X, mini_batch_Y)
            self._update_params()
            cost = self._compute_cost(A_last, mini_batch_Y, cost_func)
            J_history_batches.append(cost)
            cost_reg = self._compute_cost_reg(A_last, mini_batch_Y,cost_func) if self.reg_param is not None else cost
            if details:
                print(f"Epoch: {epoch:03d}, Batch: last, Cost: {cost_reg:.6f}")
        A_last = self._fwd_prp(X_shuffled)
        cost = self._compute_cost(A_last, Y_shuffled, cost_func)
        J_history_entire.append(cost)
        cost_reg = self._compute_cost_reg(A_last, Y_shuffled,cost_func) if self.reg_param is not None else cost
        if 'sigm' in self.act_funcs[-1]:
          predictions = self.predict_bin(X_shuffled)
        else:
          predictions = self.predict(X_shuffled)
        true_Y = np.argmax(Y_shuffled, axis=0)
        accuracy = self.get_accuracy(predictions, true_Y)
        print(f"Epoch: {epoch:03d}, Cost: {cost_reg:.6f}, accuracy: {accuracy:.4f}")
    self._plotter(J_history_entire) if plot_costs else None
    return J_history_batches, J_history_entire


  def predict(self, X):
      A_last = self._fwd_prp(X)
      predictions = np.argmax(A_last, axis = 0)
      return predictions
  def get_accuracy(self, predictions, Y):
      accuracy = 100* np.mean(predictions == Y)
      return accuracy

  def k_fold_cv(self, X, Y, k_folds, cost_func, details=True,plot_acc=True,plot_cost_vs_epoch=True,retrain=True):
    m = X.shape[1]
    indices = np.random.permutation(m)
    X_shuffled = X[:, indices]
    if plot_cost_vs_epoch:
      J_hist_list = []
    # Y = self._one_hot(Y) dont use one hot here, as the train function takes the 1d Y and does the one hot there only
    Y_shuffled = Y[indices]
    fold_size = m//k_folds
    training_accuracies = []
    testing_accuracies = []
    for fold in range(k_folds):
      print(f"Working on fold {fold+1}.....")
      test_start = fold*fold_size
      test_end = (fold+1)*fold_size if fold < k_folds - 1 else m
      self._init_params(log_it=False)
      X_train = np.concatenate((X_shuffled[:, :test_start], X_shuffled[:, test_end:]), axis = 1)
      Y_train = np.concatenate((Y_shuffled[:test_start], Y_shuffled[test_end:]))
      X_test = X_shuffled[:,test_start:test_end]
      Y_test = Y_shuffled[test_start:test_end]
      useless,J_hist = self.train(X_train, Y_train, cost_func, details,plot_costs=False)
      if plot_cost_vs_epoch:
        J_hist_list.append(J_hist)
      training_predictions = self.predict_bin(X_train) if 'sigm' in self.act_funcs[-1] else self.predict(X_train)
      testing_predictions = self.predict_bin(X_test) if 'sigm' in self.act_funcs[-1] else self.predict(X_test)
      # training_predictions = self.predict(X_train)
      # testing_predictions = self.predict(X_test)
      training_accuracy = self.get_accuracy(training_predictions, Y_train)
      print(f"The training accuracy for fold: {fold+1} is {training_accuracy:.4f}")
      testing_accuracy = self.get_accuracy(testing_predictions, Y_test)
      print(f"The testing accuracy for fold: {fold+1} is {testing_accuracy:.4f}")
      training_accuracies.append(training_accuracy)
      testing_accuracies.append(testing_accuracy)
      print(f"Fold {fold+1} Completed!")
      print(f"Starting fold {fold+2}") if fold < k_folds - 1 else None
    mean_of_training_accuracies = np.mean(training_accuracies)
    mean_of_testing_accuracies = np.mean(testing_accuracies)
    print(f"Mean of training accuracies: {mean_of_training_accuracies:.4f}")
    print(f"Mean of testing accuracies: {mean_of_testing_accuracies:.4f}")
    if plot_cost_vs_epoch:
      for fold in range(k_folds):
            plt.plot(np.arange(1,len(J_hist_list[fold])+1), J_hist_list[fold], c='r')
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.title(f"Cost vs Epochs for fold {fold+1}")
            plt.show()
    if retrain:
      print("Retraining the model on the entire dataset....")
      self._init_params(log_it=True)
      self.train(X, Y, cost_func, details,plot_costs=False)
      print("Retraining complete.")


    self._plotter_for_CV(training_accuracies,testing_accuracies) if plot_acc else None

    return training_accuracies, testing_accuracies, mean_of_training_accuracies, mean_of_testing_accuracies


  def _plotter(self, J_history):
    plt.plot(np.arange(1,len(J_history)+1), J_history, c='r')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title("Cost vs Epochs")
    plt.show()

  def _plotter_for_CV(self, training_accuracies, testing_accuracies):
    k_folds = len(training_accuracies)
    plt.plot(np.arange(1,k_folds+1), training_accuracies, c='green',label='Training Accuracy')
    plt.plot(np.arange(1,k_folds+1), testing_accuracies, c='blue',label='Testing Accuracy')
    plt.xlabel("Folds")
    plt.ylabel("Accuracies")
    plt.title("Folds vs Accuracies")
    plt.legend()
    plt.show()
  def predict_bin(self,X):
      A_last = self._fwd_prp(X)
      # predictions = np.zeros((X.shape[0],))
      # for i in range(len(X.shape[0])):
      #   predictions[i] = 1 if A_last[i] > 0.5 else 0
      predictions = (A_last > 0.5).astype(int)
      return predictions
    
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
