from __future__ import division, print_function
import numpy as np
import lasagne
import theano
import theano.tensor as T
import lasagne.layers as layers
import lasagne.nonlinearities as nl
import lasagne.objectives
import time
import copy


# TODO: save all distances calculated for the validation data
# so that this matrix can be re-used for computing auc/eer


epsilon = np.finfo(np.float32).eps


def iterate_minibatches(inputs, batchsize=1000, shuffle=False):
    """Generate mini batches from datasets
    """
    len_data = len(inputs[0])
    assert all([len(i) == len_data for i in inputs[1:]]), [len(i) for i in inputs]
    if shuffle:
        indices = np.arange(len_data)
        np.random.shuffle(indices)
    if len_data < batchsize:
        # input shorter than a single batch
        if shuffle:
            excerpt = indices
        else:
            excerpt = slice(len_data)
        yield [inp[excerpt] for inp in inputs]
    # for start_idx in range(0, len_data - batchsize + 1, batchsize):
    for start_idx in range(0, len_data, batchsize):
        if shuffle:
            excerpt = indices[start_idx:min(start_idx + batchsize, len_data)]
        else:
            excerpt = slice(start_idx, min(start_idx + batchsize, len_data))
        yield [inp[excerpt] for inp in inputs]


def train_iteration(data_train, data_val, train_fn, val_fn):
    """Generic train iteration: one full pass over the data"""
    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(data_train, 500, shuffle=True):
        train_err += train_fn(batch)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(data_val, 500, shuffle=False):
        aux = val_fn(batch)
        try:
            err, acc = aux
            val_acc += acc
        except:
            err = aux[0]
            val_acc = None
        val_err += err
        val_batches += 1

    return train_err / train_batches, val_err / val_batches


def train(data_train, data_val, train_fn, val_fn, network, max_epochs=100, patience=20, save_run=True, eval_fn=None):
    """Generic train strategy for neural networks
    (batch training, train/val sets, patience)

    Trains a neural network according to some data (list of inputs, targets)
    and a train function and an eval function on that data"""
    print("training...")

    run = []
    best_model = None
    best_epoch = None
    if patience <= 0:
        patience = max_epochs
    patience_val = 0
    best_val = None

    for epoch in range(max_epochs):
        start_time = time.time()
        train_err, val_err = train_iteration(data_train, data_val,
                                             train_fn, val_fn)

        run.append(layers.get_all_param_values(network))
        if np.isnan(val_err) or np.isnan(train_err):
            print("Train error or validation error is NaN, "
                  "stopping now.")
            break
        # Calculating patience
        if best_val == None or val_err < best_val:
            best_val = val_err
            patience_val = 0
            best_model = layers.get_all_param_values(network)
            best_epoch = epoch
        else:
            patience_val += 1
            if patience_val > patience:
                print("No improvements after {} iterations, "
                      "stopping now".format(patience))
                break

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        try:
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
        except:
            if eval_fn != None:
                acc = eval_fn(*data_val)
                print("  validation accuracy:\t\t{:.2f} %".format(acc))

    return best_model, best_epoch, run


class Classifier_Nnet(object):
    """Build a standard deep neural network for classification
    """
    def __init__(self, dims, dropouts=None, input_var=None):
        assert len(dims) >= 3, 'Not enough dimmensions'
        if dropouts != None:
            assert len(dropouts) == len(dims) - 1
        else:
            dropouts = [0] * (len(dims) - 1)
        self.input_var = input_var
        if input_var == None:
            self.input_var = T.matrix('inputs')
        self.target_var = T.ivector('targets')

        # input layer
        network = layers.InputLayer((None, dims[0]), input_var=self.input_var)
        if dropouts[0]:
            network = layers.DropoutLayer(network, p=dropouts[0])
        # hidden layers
        for dim, dropout in zip(dims[1:-1], dropouts[1:]):
            network = layers.DenseLayer(network, num_units=dim,
                                        W=lasagne.init.GlorotUniform())
            if dropout:
                network = layers.DropoutLayer(network, p=dropout)
        # output layer
        network = layers.DenseLayer(network, num_units=dims[-1],
                                  nonlinearity=nl.softmax)
        self.network = network

        # util functions, completely stolen from Lasagne example
        self.prediction = layers.get_output(network)
        self.loss = lasagne.objectives.categorical_crossentropy(
            self.prediction, self.target_var).mean()
        self.params = layers.get_all_params(network, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.loss, self.params, learning_rate=0.01, momentum=0.9)
        # if non-determnistic:
        self.test_prediction = layers.get_output(network, deterministic=True)
        self.test_loss = lasagne.objectives.categorical_crossentropy(
            self.test_prediction, self.target_var).mean()
        self.test_acc = T.mean(
            T.eq(T.argmax(self.test_prediction, axis=1), self.target_var),
            dtype=theano.config.floatX)

        self.train_fn = theano.function([self.input_var, self.target_var],
                                        self.loss, updates=self.updates)
        self.val_fn = theano.function([self.input_var, self.target_var],
                                      [self.test_loss, self.test_acc])
        self.eval_fn = theano.function([self.input_var], [self.test_prediction])
        self.acc_fn = theano.function([self.input_var, self.target_var], self.test_acc)


    def train(self, X_train, y_train, X_val, y_val, max_epochs=500, patience=20):
        def train_batch(batch_data):
            return self.train_fn(*batch_data)
        def val_batch(batch_data):
            return self.val_fn(*batch_data)

        train([X_train, y_train], [X_val, y_val], train_batch, val_batch,
              self.network, max_epochs=max_epochs, patience=patience)

    def evaluate(self, X_test):
        for batch in iterate_minibatches([X_test], 500, shuffle=False):
            return self.test_prediction(*batch)

    def score(self, X_test, Y_test):
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))


class ABnet(object):
    """Siamese neural network
    """
    def __init__(self, dims, nonlinearities=None, dropouts=None,
                 update_fn=None, batch_norm=False,
                 loss_type='cosine_margin', margin=0.8):
        """Initialize a Siamese neural network

        Parameters:
        -----------
        update_fn: theano function with 2 arguments (loss, params)
            Update scheme, default to adadelta
        batch_norm: bool
            Do batch normalisation on first layer, default to false
        """
        assert len(dims) >= 3, 'Not enough dimmensions'
        if dropouts != None:
            dropouts = copy.copy(dropouts)
            assert len(dropouts) == len(dims) - 1
            dropouts.append(0)
        else:
            dropouts = [0] * len(dims)
        if nonlinearities==None:
            nonlinearities = [nl.sigmoid] * (len(dims) -1)
        else:
            assert len(nonlinearities) == len(dims) - 1
        if update_fn == None:
            update_fn = lasagne.updates.adadelta
        self.input_var1 = T.matrix('inputs1')
        self.input_var2 = T.matrix('inputs2')
        self.target_var = T.ivector('targets')
        # input layer
        network1 = layers.InputLayer((None, dims[0]), input_var=self.input_var1)
        network2 = layers.InputLayer((None, dims[0]), input_var=self.input_var2)
        if dropouts[0]:
            network1 = layers.DropoutLayer(network1, p=dropouts[0])
            network2 = layers.DropoutLayer(network2, p=dropouts[0])
        # hidden layers
        for dim, dropout, nonlin in zip(dims[1:], dropouts[1:], nonlinearities):
            network1 = layers.DenseLayer(network1, num_units=dim,
                                         W=lasagne.init.GlorotUniform(),
                                         nonlinearity=nonlin)
            network2 = layers.DenseLayer(network2, num_units=dim,
                                         W=network1.W, b=network1.b,
                                         nonlinearity=nonlin)
            if batch_norm:
                network1 = layers.batch_norm(network1)
                network2 = layers.batch_norm(network2)
            if dropout:
                network1 = layers.DropoutLayer(network1, p=dropout)
                network2 = layers.DropoutLayer(network2, p=dropout)
        self.network = [network1, network2]
        self.params = layers.get_all_params(network1, trainable=True)

        # util functions, completely stolen from Lasagne example
        self.prediction1 = layers.get_output(network1)
        self.prediction2 = layers.get_output(network2)
        # if non-determnistic:
        self.test_prediction1 = layers.get_output(network1, deterministic=True)
        self.test_prediction2 = layers.get_output(network2, deterministic=True)

        self.change_loss(loss_type, margin)
        self.change_update(update_fn)

    def change_loss(self, loss_type='cosine_margin', margin=0.8):
        self.loss = loss_fn(self.prediction1, self.prediction2,
                            self.target_var, loss_type, margin)
        self.loss_eval = loss_fn(self.test_prediction1, self.test_prediction2,
                                 self.target_var, loss_type, margin,
                                 return_similarities=True)
        self.test_loss = loss_fn(self.test_prediction1, self.test_prediction2,
                                 self.target_var, loss_type, margin)
        self.change_update()

    def change_update(self, update_fn=lasagne.updates.adadelta):
        self.updates = update_fn(self.loss, self.params)
        self.train_fn = theano.function(
            [self.input_var1, self.input_var2, self.target_var],
            self.loss, updates=self.updates)
        self.train_and_eval_fn = theano.function(
            [self.input_var1, self.input_var2, self.target_var],
            self.loss, updates=self.updates)
        self.val_fn = theano.function(
            [self.input_var1, self.input_var2, self.target_var],
            [self.test_loss])
        self.eval_fn = theano.function(
            [self.input_var1], self.test_prediction1)

    def train(self, X_train1, X_train2, y_train, X_val1, X_val2, y_val, max_epochs=500, patience=20):
        def train_batch(batch_data):
            return self.train_fn(*batch_data)
        def val_batch(batch_data):
            return self.val_fn(*batch_data)

        best_weights, best_epoch, run = train(
            [X_train1, X_train2, y_train], [X_val1, X_val2, y_val],
            train_batch, val_batch,
            self.network, max_epochs=max_epochs, patience=patience)
        layers.set_all_param_values(self.network, best_weights)
        return run, epoch

    def evaluate(self, X_test):
        embs = []
        for batch in iterate_minibatches([X_test], 500, shuffle=False):
            inputs = batch[0]
            emb = self.eval_fn(inputs)
            embs.append(emb)
        if len(embs) > 1:
            return np.concatenate(embs)
        else:
            return embs[0]

    def eer(self, X1, X2, y):
        """Estimation of the equal error rate
        Returns % of correctly classified data with optimal threshold
        """
        distances = self._distances(X1, X2)
        distances_same = distances[y==1]
        distances_diff = distances[y==0]
        # finding optimal threshold
        threshold = np.mean(distances)
        t_min = np.min(distances)
        t_max = np.max(distances)
        misclassified_diff = np.sum(distances_diff < threshold)
        misclassified_same = np.sum(distances_same > threshold)
        while misclassified_diff - misclassified_same > 1:
            misclassified_diff = np.sum(distances_diff < threshold)
            misclassified_same = np.sum(distances_same > threshold)
            if misclassified_diff > misclassified_same:
                t_max = threshold
                threshold = (threshold + t_min) / 2
            else:
                t_min = threshold
                threshold = (threshold + t_max) / 2

        error_rate = (misclassified_diff + misclassified_same) / y.shape[0]
        return (1 - error_rate) * 100

    def abx(self, X1, X2, y):
        """Discriminability estimation"""
        raise NotImplementedError

    def _distances(self, X1, X2):
        distances = []
        cos_theano = cos_sim(self.prediction1, self.prediction2)
        cos_fn = theano.function([self.input_var1, self.input_var2],
                                 cos_theano)
        for batch in iterate_minibatches([X1, X2]):
            distances.append((1 - cos_fn(*batch)) / 2)
        if len(distances) > 1:
            distances = np.concatenate(distances)
        else:
            distances = distances[0]
        return distances

    def auc(self, X1, X2, y):
        """Area under the ROC curve
        returns (auc * 2 - 1) * 100 (effective score in percent,
        chance at 0%)"""
        nbins = 100
        distances = self._distances(X1, X2)
        distances_same = distances[y==1]
        distances_diff = distances[y==0]
        thresholds = np.linspace(np.min(distances), np.max(distances), nbins)
        misclassified_diff, misclassified_same = np.empty((nbins,), dtype=float), np.empty((nbins,), dtype=float)
        for idx, threshold in enumerate(thresholds):
            misclassified_diff[idx] = float(np.sum(distances_diff < threshold)) / distances_diff.shape[0]
            misclassified_same[idx] = float(np.sum(distances_same > threshold)) / distances_same.shape[0]
        area = 0.
        for i in range(nbins - 1):
            x0 = misclassified_diff[i]
            x1 = misclassified_diff[i+1]
            y0 = misclassified_same[i]
            y1 = misclassified_same[i+1]
            area += (y0 + y1) * (x1 - x0) / 2
            # print(area)
        area = 1 - area
        return ((area * 2) - 1) * 100

    # def pretrain_AE(self, AE=1, nonlinearity=None):
    #     """Pretrain with autoencoder

    #     Parameters:
    #     ----------
    #     AE: str or int, number of additional hidden layers before output,
    #         if AE='mirror', it will use n_hid - 1 layers (n_hid being the
    #         number of hidden layers)
    #     nonlinearity: callable, if None, the non linearity of the penultimate
    #         layer will be used (unless 'mirror' is chosen, in which case the
    #         non linearities are chosen in mirror"""
    #     ae_network = self.network[0]
    #     if nonlinearity == None:
    #         #TODO get the correct layer everytime
    #         nonlinearity = ae_network.inputlayer.nonlinearity
    #     if AE isinstance(str) and AE == 'mirror':
    #         raise NotImplementedError
    #     else:
    #         for i in range(AE):
    #             raise NotImplementedError
    #             ae_network = layers.DenseLayer(
    #                 ae_network, num_units=dim,
    #                 W=lasagne.init.GlorotUniform(),
    #                 nonlinearity=nonlinearity)
    #     #dim = inputdim
    #     ae_network = layers.DenseLayer(
    #                 ae_network, num_units=dim,
    #                 W=lasagne.init.GlorotUniform(),
    #                 nonlinearity=nonlinearity)

    #     ae_params = layers.get_all_params(ae_network, trainable=True)
    #     ae_prediction = layers.get_output(ae_network)
    #     ae_loss = T.mean((ae_prediction - self.input_var1).norm(2, axis=-1))
    #     ae_updates = lasagne.updates.adadelta(ae_loss, ae_params)
    #     ae_train = theano.function(self.input_var1, ae_loss,
    #                                updates=ae_updates)
    #     ae_val = theano.function(self.input_var1, ae_loss)

    #     def train_batch(batch_data):
    #         return self.train_fn(*batch_data)
    #     def val_batch(batch_data):
    #         return self.val_fn(*batch_data)
    #     best_weights, run = train(
    #         [X_train], [X_val],
    #         train_batch, val_batch,
    #         ae_network, max_epochs=100, patience=10)
    #     layers.set_all_param_values(self.ae_network, best_weights)

def cos_sim(vector1, vector2):
    res = T.sum(vector1 * vector2, axis=-1) / (vector1.norm(2, axis=-1) * vector2.norm(2, axis=-1))
    return res

def loss_fn(prediction1, prediction2, targets, loss='cosine_margin', margin=0.15,
            return_similarities=False):
    # Cosine similarity:
    cos_sim = (
        T.sum(prediction1 * prediction2, axis=-1) /
        (prediction1.norm(2, axis=-1) * prediction2.norm(2, axis=-1)))
    # eucl_dist = (prediction1 - prediction2).norm(2, axis=-1) + epsilon

    # # (more stable) cosine similarity:
    # cos_sim = T.sum(self.prediction1 * self.prediction2, axis=-1) / (self.prediction1.norm(2, axis=-1) * self.prediction2.norm(2, axis=-1) + epsilon)

    # # cos cos^2 loss function:
    # cos2_sim_cost = T.switch(self.target_var, 1.-T.sqrt(self.cos_sim), self.cos_sim ** 2)
    # self.loss = T.mean(cos2_sim_cost)

    if loss == 'cosine_margin':
        # part linear loss function:
        x0 = 1 - margin
        # cos_sim_same = T.switch(cos_sim>0.9, 0, 1 - (cos_sim / x0))
        cos_sim_diff = T.switch(cos_sim < x0, 0, (cos_sim - x0) / (1 - x0))
        cos_sim_same = 1. - cos_sim
        # cos_sim_diff = T.switch(cos_sim<0.9, 0, cos_sim - x0)
    elif loss == 'cosine_bounded':
        x0 = 1 - margin
        cos_sim_same = T.switch(cos_sim < x0, 1, (1 - cos_sim) / (1 - x0))
        cos_sim_diff = T.switch(cos_sim < x0, 0, (cos_sim - x0) / (1 - x0))
    elif loss == 'coscos2':
        cos_sim_same = 1.-T.sqrt(cos_sim)
        cos_sim_diff = cos_sim ** 2

    # elif loss == 'euclidian_margin':
    #     x0 = margin
    #     cos_sim_diff = T.switch(eucl_dist > x0, 0, (1 - eucl_dist - x0) / (1 - x0))
    #     cos_sim_same = eucl_dist

    if return_similarities:
        return T.mean(T.switch(targets, cos_sim_same, cos_sim_diff)), cos_sim
    else:
        return T.mean(T.switch(targets, cos_sim_same, cos_sim_diff))
