import numpy as np
import lasagne
import theano
import theano.tensor as T
import lasagne.layers as layers
import lasagne.nonlinearities as nl
import lasagne.objectives
import time
import copy

epsilon = np.finfo(np.float32).eps

def iterate_minibatches(inputs, batchsize=1000, shuffle=False):
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
    """Generic train iteration: one pass over the data"""
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


def train(data_train, data_val, train_fn, val_fn, network, max_epochs=100, patience=20, save_run=True):
    """Generic train strategy for neural networks
    (batch training, train/val sets, patience)

    Trains a neural network according to some data (list of inputs, targets)
    and a train function and an eval function on that data"""

    run = []
    best_model = None
    if patience <= 0:
        patience = max_epochs
    patiance_val = 0
    best_val = None
    
    for epoch in range(max_epochs):
        start_time = time.time()
        train_err, val_err = train_iteration(data_train, data_val,
                                             train_fn, val_fn)

        run.append([np.array(x) for x in layers.get_all_param_values(
            network, regularizable=True)])
        if np.isnan(val_err) or np.isnan(train_err):
            print("Train error or validation error is NaN, "
                  "stopping now.")
            break
        # Calculating patience
        if best_val == None or val_err < best_val:
            best_val = val_err
            patience_val = 0
            best_model = [np.array(x) for x in layers.get_all_param_values(
                network, regularizable=True)]
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
            pass

    return best_model, run


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

    def evaluate(X_test):
        for batch in iterate_minibatches([X_test], 500, shuffle=False):
            return self.test_prediction(*batch)

    def score(self, X_test, Y_test):
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
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
                 update_fn=lasagne.updates.adadelta):
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
            if dropout:
                network1 = layers.DropoutLayer(network1, p=dropout)
                network2 = layers.DropoutLayer(network2, p=dropout)
        self.network = [network1, network2]

        # util functions, completely stolen from Lasagne example
        self.prediction1 = layers.get_output(network1)
        self.prediction2 = layers.get_output(network2)

        def loss_fn(prediction1, prediction2, targets):
            # Cosine similarity:
            cos_sim = (
                T.sum(prediction1 * prediction2, axis=-1) /
                (prediction1.norm(2, axis=-1) * prediction2.norm(2, axis=-1)))

            # # (more stable) cosine similarity:
            # cos_sim = T.sum(self.prediction1 * self.prediction2, axis=-1) / (self.prediction1.norm(2, axis=-1) * self.prediction2.norm(2, axis=-1) + epsilon)

            # # cos cos^2 loss function:
            # cos2_sim_cost = T.switch(self.target_var, 1.-T.sqrt(self.cos_sim), self.cos_sim ** 2)
            # self.loss = T.mean(cos2_sim_cost)

            # part linear loss function:
            x0 = 0.9
            # cos_sim_same = T.switch(cos_sim>0.9, 0, 1 - (cos_sim / x0))
            # cos_sim_diff = T.switch(cos_sim<0.9, 0, (cos_sim - x0) / (1 - x0))
            cos_sim_same = 1. - cos_sim
            cos_sim_diff = T.switch(cos_sim<0.9, 0, (cos_sim - x0) / (1 - x0))
            return T.mean(T.switch(targets, cos_sim_same, cos_sim_diff))

        self.loss = loss_fn(self.prediction1, self.prediction2, self.target_var)
        self.params = layers.get_all_params(network1, trainable=True)
        self.updates = update_fn(self.loss, self.params)
        # if non-determnistic:
        self.test_prediction1 = layers.get_output(network1, deterministic=True)
        self.test_prediction2 = layers.get_output(network2, deterministic=True)
        self.test_loss = loss_fn(self.test_prediction1, self.test_prediction2,
                                 self.target_var)

        self.train_fn = theano.function(
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

        best_weights, run = train(
            [X_train1, X_train2, y_train], [X_val1, X_val2, y_val],
            train_batch, val_batch,
            self.network, max_epochs=max_epochs, patience=patience)
        layers.set_all_param_values(self.network, best_weights, regularizable=True)
        return run

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
