import abnet2
import lasagne
import lasagne.layers as layers
import h5py
import ABXpy.h5tools.h52np as h52np
import ABXpy.h5tools.np2h5 as np2h5
import os
import shutil
import lasagne.nonlinearities as nl
import time
import sample_data
import numpy as np


def tryremove(f):
    try:
        os.remove(f)
    except:
        pass

    
def run(train_words, val_words, alignement_features, input_features,
        name='abnet', dim=40, nframes=7):
    data_train = sample_data.read_word_clusters_file(train_words)
    data_test = sample_data.read_word_clusters_file(val_words)
    data = [data_train, data_test]

    layer_size = 500
    n_layers_init = 2
    n_layers_end = 6
    n_layers = n_layers_init
    architecture = [[dim*nframes] + [layer_size]*n_layers_init + [39],
                    [nl.rectify]*n_layers_init + [nl.linear],
                    [0] + [0.2]*n_layers_init]
    nnet = abnet2.ABnet(*architecture, loss_type='cosine_margin', margin=0.5)
    features_getter = sample_data.FeaturesAPI_cached(alignement_features)
    input_features_getter = sample_data.FeaturesAPI_cached(input_features)


    # Utility functions
    def train_fn(batch_data):
        return nnet.train_fn(*batch_data)
    def val_fn(batch_data):
        return nnet.val_fn(*batch_data)
    def train(data, train_fn, val_fn, network, max_epochs=4000, patience=100):
        (train_words, train_clusters), (test_words, test_clusters) = data
        run = []
        best_model = None
        if patience <= 0:
            patience = max_epochs
        patience_val = 0
        best_val = None

        for epoch in range(max_epochs):
            data_train = sample_data.generate_abnet_batch(
                train_words, train_clusters, epoch, features_getter,
                input_features_getter, return_indexes=False)
            data_val = sample_data.generate_abnet_batch(
                test_words, test_clusters, epoch, features_getter,
                input_features_getter, return_indexes=False)
            start_time = time.time()
            train_err, val_err = abnet2.train_iteration(
                data_train, data_val, train_fn, val_fn)
            if epoch % 20 == 0:
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
            acc = nnet.eer(*data_val)
            print("  score eer:\t\t{:.2f} %".format(acc))
            auc = nnet.auc(*data_val)
            print("  score auc:\t\t{:.2f} %".format(auc))
        return best_model, run


    # Training...
    for i in range(n_layers_end - n_layers_init):
        best_model, run = train(data, train_fn, val_fn, nnet.network)
        # saving model
        weights = best_model
        weights_file = '{}_{}_best_model.npz'.format(name, i)
        run_file = '{}_{}_run.h5'.format(name, i)

        tryremove(weights_file)
        tryremove(run_file)
        np.savez(weights_file, weights)

        for epoch, weights in enumerate(run):
            h5py.File(run_file).create_group(str(epoch * 2))
            for i, w in enumerate(weights[::2]):
                h5py.File(run_file)[str(epoch * 2)].create_dataset(str(i), data=w)

        # adding layer
        if i < n_layers_end - n_layers_init - 1:
            n_layers += 1
            W = lasagne.init.GlorotUniform().sample((layer_size, layer_size))
            u = np.empty((layer_size,), dtype=np.float32)
            architecture = [[dim*nframes] + [layer_size]*n_layers + [39],
                            [nl.rectify]*n_layers + [nl.linear],
                            [0] + [0.2, 0.2, 0.2, 0.2]*n_layers]
            init_weights = best_model[:-2] + [W, u] + best_model[-2:]
            nnet = abnet2.ABnet(*architecture, loss_type='cosine_margin', margin=0.5)
            layers.set_all_param_values(nnet.network, init_weights)


    layers.set_all_param_values(nnet.network, best_model)
    h5features_file = name + '_embeddings.h5f'
    try:
        tryremove(h5features_file)
        shutil.copy(input_features, h5features_file)
        del h5py.File(h5features_file)['features']['features']
        transform = nnet.evaluate
        embedding_size = architecture[0][-1]

        with h52np.H52NP(input_features) as f_in, \
             np2h5.NP2H5(h5features_file) as f_out:
            inp = f_in.add_dataset('features', 'features', buf_size=10000)
            out = f_out.add_dataset(
                'features', 'features', buf_size=10000,
                n_rows=inp.n_rows, n_columns=embedding_size,
                item_type=np.float32)
            for X in inp:
                X = X.astype(np.float32)
                emb_wrd = transform(X)
                out.write(emb_wrd)
    except:
        tryremove(h5features_file)
        raise
