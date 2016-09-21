from __future__ import division, print_function
import numpy as np
# import gmpy
from collections import namedtuple
from dtw import DTW
from ABXpy.distances.metrics.cosine import cosine_distance
from time import time
import h5py
import h5features


"""Sampling module

Sampling scheme:
- Uniformly sample from the list of all possible words
- For each selected word, sample a second word uniformly over
the set of words with same label

TODO: respect speaker proportion in first sample, respect same
speaker ratio and speaker proportion in second sample
TODO: class for list of words (for word keeping index)

Takes as input a word cluster file (constructed here with the
get_clusters function.



WORD_CLUSTER_FILE:

word1 n_phones
file start end
file start end
...

word2 n_phones
...

note:
'file' correspond to the item in the h5features file.
'start' and 'end' are in seconds.

A 'realisation' refers to the production of a certain word in the corpus
It is stored in a "Word" object.
"""

# FB_FILE = 'fb.h5f'
# FB_MVN_FILE = 'fb_mvn.h5f'
# FB_MVN_STACKED_FILE = 'fb_mvn_stacked.h5f'
N_FRAMES = 7
MARGIN = N_FRAMES // 2
WORD_CLUSTERS_FILE = 'words_gold.txt'

Word = namedtuple('Word', ['value', 'intvl', 'speaker', 'index'])
# index refer to its index in the list of words with the same label
Interval = namedtuple('Interval', ['file', 'start', 'end'])


def get_clusters():
    spk_list = set((np.loadtxt('../zerospeech2015/tde/bin/resources/english.split',
                           dtype='S6', delimiter=' ', usecols=[0])))
    spk_list.remove('s0103a')
    annotations_folder = '/fhgfs/bootphon/data/derived_data/BUCKEYE_revised/data/annotations/wrd/'
    dictionnary = {}
    #import glob
    #for f in glob.glob(annotations_folder + '*.wrd'):
    for spk in spk_list:
        f = annotations_folder + spk + '.wrd'
        with open(f) as fin:
            for linenb, line in enumerate(fin):
                splitted = line.split('\t')
                annotation = splitted[2].split('; ')
                if annotation[0][0] != '<' and annotation[0] not in dictionnary:
                    try:
                        dictionnary[annotation[0]] = annotation[1].split()
                    except:  # annotation problem in the new buckeye
                        pass
    dictionnary.update({w.replace('\'', '').replace('.', ''): phns
                        for w, phns in dictionnary.iteritems()})
    dictionnary['craze'] = ['k', 'r', 'ey', 'z']
    n_phones = {word: len(phns) for word, phns in dictionnary.iteritems()}

    gold_words = []
    with open('/fhgfs/bootphon/scratch/roland/zerospeech2015/tde/bin/resources/english.wrd') as fin:
        for line in fin:
            splitted = line.strip().split()
            word = splitted[3]
            if word == 'SIL' or word == 'SPN':
                continue
            start, end = map(float, splitted[1:3])
            gold_words.append((word, splitted[0], start, end))
    clustered_gold_words = {}
    for w in gold_words:
        try:
            clustered_gold_words[w[0]].append(w[1:])
        except KeyError:
            clustered_gold_words[w[0]] = [n_phones[w[0]], w[1:]]
    with open(WORD_CLUSTERS_FILE, 'w') as fout:
        for word, tokens in clustered_gold_words.iteritems():
            fout.write(word + ' ' + str(tokens[0]) + '\n')
            for token in tokens[1:]:
                fout.write(' '.join([str(field) for field in token]) + '\n')
            fout.write('\n')


def get_speaker(fname):
    return fname[:3]


def read_word_clusters_file(word_clusters_file, min_number_phones=4):
    """Read a word clusters file and output a list of words, a
    dictionnary of clusters and the number of possible pairs
    
    Parameters:
    ----------
    word_clusters_file: str, filename of the words file (see module doc)
    min_number_phones: int, minimal number of phones for a word to be
        considered

    Returns:
    words: list of all realisations
    clusters: dictionnary containing the possible realisations for each word
    """
    clusters = {}
    words = []
    new_class = True
    current_class = None
    current_words = []
    # count = 0
    with open(word_clusters_file) as fin:
        for line in fin:
            if new_class:
                assert current_class == None
                try:
                    current_class, n_phones = line.strip().split()
                    n_phones = int(n_phones)
                except:
                    current_class = line.strip()
                    n_phones = min_number_phones
                new_class = False
            else:
                if line == '\n':
                    # end of cluster, wrap up
                    if ((len(current_words) >= 2 and
                         n_phones >= min_number_phones)):
                        # enough words for a pair
                        clusters[current_class] = current_words
                        words.extend(current_words)
                        # count += gmpy2.comb(2, len(current_words))
                    current_class = None
                    new_class = True
                    current_words = []
                else:
                    # add word to cluster
                    content = line.strip().split()
                    fname = content[0]
                    intvl = Interval(content[0], *map(float, content[1:]))
                    current_words.append(Word(
                        current_class, intvl, get_speaker(fname),
                        len(current_words) - 1))
    return words, clusters  #, long(count)


def sample_test_set(words, clusters, n_pairs):
    """Sampling test set and removing those words from the available
    ones

    NOTIMPLEMENTED, may not be useful since the test set should be a separate
    set anyway"""
    pairs = []
    indexes_1 = np.random.choice(len(words), n_pairs, replace=False)
    for i1 in indexes_1:
        word1 = words[i1]
        label = word1.value
        same_words = clusters[label]
        i2 = np.random.choice(len(same_words) - 1)
        if i2 >= word1.index:
            i2 += 1
        word2 = same_words[i2]
        pairs.append((word1, word2))
    #TODO: delete the words and update index accordingly
    # # innefficient:
    # indexes = sorted(indexes_1 + indexes_2)
    # shift = 0
    # for i in indexes_1:
    #     words.pop(i-shift)
    #     shift += 1
    raise NotImplementedError


def sample_train_batch(words, clusters, n_pairs=None, hard=False):
    """Sampling a train batch.

    Sample pairs of realisations with the same label and pairs of realisations
    with a different label. Samples as many "same" pairs as "diff" pairs.

    Parameters:
    words: list, list of realisations. Type: [Word]
    clusters: dict, dictionnary assiocating a word label to its realasations
        int the corpus. Type: {str: [Word]}
    (output of read_word_clusters_file)
    n_pairs: int. Number of "same" and "diff" pairs to sample
    hard: bool, will sample "same" pairs across speaker and "diff" pairs within
        speaker only

    Returns:
    pairs_same: list. List of pairs of realisations with the same label.
        Type: [(Word, Word)]
    pairs_diff:list. List of pairs of realisations with a different label.
        Type: [(Word, Word)]
    """
    if n_pairs == None:
        n_pairs = len(words)
    # sampling "same" words
    pairs_same = []
    indexes_1 = np.random.choice(len(words), n_pairs, replace=False)
    for i1 in indexes_1:
        word1 = words[i1]
        label = word1.value
        same_words = clusters[label]
        if hard:
            same_words = [w for w in clusters[label]
                          if w.speaker != word1.speaker]
            if len(same_words) <= 1:
                continue
        i2 = np.random.choice(len(same_words) - 1)
        if i2 >= word1.index:
            i2 += 1
        word2 = same_words[i2]
        pairs_same.append((word1, word2))
    # sampling "different" words
    pairs_diff = []
    indexes1 = np.random.choice(len(words), n_pairs, replace=False)
    indexes2 = np.random.choice(len(words), n_pairs, replace=False)
    if hard:
        speaker = {}
        for w in words:
            try:
                speaker[w.speaker].append(w)
            except KeyError:
                speaker[w.speaker] = [w]
    for i1, i2 in zip(indexes1, indexes2):
        if hard:
            diff_words = speaker[words[i1].speaker]
            i2 = np.random.choice(len(diff_words))
            while words[i1].value == diff_words[i2].value:
                i2 = np.random.choice(len(diff_words))
            pairs_diff.append((words[i1], diff_words[i2]))
        else:
            while words[i1].value == words[i2].value:
                i2 = np.random.choice(len(words))
            pairs_diff.append((words[i1], words[i2]))
    return pairs_same, pairs_diff


import cPickle
class MemoizeMutable:
    """Memoize(fn) - an instance which acts like fn but memoizes its arguments
       Will work on functions with mutable arguments (slower than Memoize)
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args, **kwargs):
        str = cPickle.dumps(args)
        if not self.memo.has_key(str):
            self.memo[str] = self.fn(*args)
        return self.memo[str]


class FeaturesAPI_cached:
    """h5features API with caching the features for fast access,
    assuming 25ms windows, 10ms step"""
    def __init__(self, feature_file):
        """Cache in the features"""
        self.feature_file = feature_file
        dset = h5py.File(feature_file).keys()[0]
        self.files = h5py.File(feature_file)[dset]['items']
        self.files_index = h5py.File(feature_file)[dset]['index']
        self.features = {}
        for f in self.files:
            self.features[f] = self.get_features_from_file(f)

    def get_features_from_file(self, fileid):
        """return the features associated to a file"""
        return h5features.read(
            self.feature_file, from_item=fileid)[1][fileid]

    def get_features(self, segment):
        fileid, start, end = segment
        feats = self.features[fileid]
        start_frame, end_frame = map(
            lambda x : int(x * 100 - 1.25), [start, end])
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, feats.shape[0])
        return feats[start_frame:end_frame]

    def get_indexes(self, segment):
        fileid, start, end = segment
        feats = self.features[fileid]
        start_frame, end_frame = map(
            lambda x : int(x * 100 - 1.25), [start, end])
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, feats.shape[0])
        aux = self.files.index(fileid)
        file_idx = self.file_index[aux-1] + 1
        # aux = self.index['files'].index(fileid)
        # file_idx = self.index['file_index'][aux-1] + 1
        if aux == 0:
            file_idx = 0
        return np.arange(file_idx+start_frame, file_idx+end_frame, dtype=np.int64)


def get_same_data(pairs_list, mfccs_file, stacked_fbanks_file,
                  features_getter=None, stacked_features_getter=None,
                  return_indexes=True):
    """Align pairs of words with DTW and returns indexes or features.

    The indexes returned correspond to indexes in the stacked fbanks
    features file. Otherwise it will return stacked filterbanks features.
    
    Parameters:
    ----------
    pairs_list: list, list of pairs of words
    mfccs_file: str, path to the h5features containing the mfccs (or
        the features with which the words will be aligned)
    stacked_fbanks_file: str, path to the h5features containing the
        stacked fbanks (or the features input to the abnet)
    return_indexes: bool, True will return indexes of the stacked fbanks
        features file. Otherwise it will directly return features.

    Returns:
    -------
    X1, X2: numpy array. Indexes of features. X1[0] and X2[0] are aligned
        frames from the same word. The ABnet must learn X1 == X2
    """
    if features_getter == None:
        features_getter = FeaturesAPI_cached(mfccs_file)
    if stacked_features_getter == None:
        stacked_features_getter = FeaturesAPI_cached(stacked_fbanks_file)

    if return_indexes:
        get_function = stacked_features_getter.get_indexes
    else:
        get_function = stacked_features_getter.get_features

    X1 = []
    X2 = []

    def get_dtw_alignement(intvl1, intvl2):
        feats1, feats2 = map(
            features_getter.get_features, [intvl1, intvl2])
        distance_array = cosine_distance(feats1, feats2)
        try:
            cost, _, paths = DTW(feats1, feats2, return_alignment=True,
                                 dist_array=distance_array)
        except:
            print('\n\n'.join(str(x) for x in [
                feats1.shape, feats2.shape, distance_array.shape,
                intvl1, intvl2]))
        path1, path2 = paths[1:]
        assert len(path1) == len(path2)
        return path1, path2
    # get_dtw_alignement = MemoizeMutable(get_dtw_alignement)

    for word1, word2 in pairs_list:
        intvl1, intvl2 = map(lambda word: word.intvl, [word1, word2])
        path1, path2 = get_dtw_alignement(intvl1, intvl2)
        #TODO: get actual index and re-use it here
        stacked_indexes1, stacked_indexes2 = map(
            get_function, [intvl1, intvl2])
        X1.append(stacked_indexes1[path1])
        X2.append(stacked_indexes2[path2])
    X1, X2 = map(np.concatenate, [X1, X2])
    return X1, X2


def get_diff_data(pairs_list, stacked_fbanks_file,
                  features_getter=None, return_indexes=True):
    """'Align' pairs of words and returns indexes or features.

    The indexes returned correspond to indexes in the stacked fbanks
    features file
    Otherwise it will return stacked filterbanks features

    Alignement here is just taking the diagonal.
    
    Parameters:
    ----------
    pairs_list: list, list of pairs of words
    stacked_fbanks_file: str, path to the h5features containing the
        stacked fbanks (or the features input to the abnet)
    return_indexes: bool, True will return indexes of the stacked fbanks
        features file. Otherwise it will directly return features.

    Returns:
    -------
    X1, X2: numpy array. Indexes of features. X1[0] and X2[0] are aligned
        frames from different words. The ABnet must learn X1 != X2.
    """
    X1 = []
    X2 = []
    if features_getter == None:
        features_getter = FeaturesAPI_cached(stacked_fbanks_file)

    if return_indexes:
        get_function = features_getter.get_indexes
    else:
        get_function = features_getter.get_features

    for word1, word2 in pairs_list:
        indexes1, indexes2 = map(
            lambda word: get_function(word.intvl),
            [word1, word2])
        length = min(len(indexes1), len(indexes2))
        X1.append(indexes1[:length])
        X2.append(indexes2[:length])
    X1, X2 = map(np.concatenate, [X1, X2])
    return X1, X2


def generate_abnet_batch(words, clusters, iteration,
                         features_getter, stacked_features_getter,
                         return_indexes=True):
    """Generate a bath a "same" and "different" examples.

    Parameters:
    ----------
    words: list, list of realisations. Type: [Word]
    clusters: dict, dictionnary assiocating a word label to its realasations
        int the corpus. Type: {str: [Word]}
    (output of read_word_clusters_file)
    iteration: int, iteration number.
    """
    pairs_same, pairs_diff = sample_train_batch(words, clusters, hard=False)
    X1_same, X2_same = get_same_data(
        pairs_same, None, None, features_getter,
        stacked_features_getter, return_indexes=return_indexes)
    X1_diff, X2_diff = get_diff_data(
        pairs_diff, None, stacked_features_getter,
        return_indexes=return_indexes)
    y_same = np.ones(X1_same.shape[0], dtype=np.int8)
    y_diff = np.zeros(X1_diff.shape[0], dtype=np.int8)
    X1, X2, y = map(np.concatenate, [
        [X1_same, X1_diff], [X2_same, X2_diff], [y_same, y_diff]])

    order = np.random.permutation(y.shape[0])  # use sklearn shuffle instead ?
    X1, X2, y = X1[order], X2[order], y[order]
    return X1, X2, y


def generate_and_save_abnet_batch(
        words, clusters, iteration, output_file,
        features_getter, stacked_features_getter,
        synchronize=None):
    """Generate a bath a "same" and "different" examples.

    Parameters:
    ----------
    words: list, list of realisations. Type: [Word]
    clusters: dict, dictionnary assiocating a word label to its realasations
        int the corpus. Type: {str: [Word]}
    (output of read_word_clusters_file)
    iteration: int, iteration number.
    output_file: hdf5 file were to write the data
    synchronize: optional file lock on the hdf5 file. Useful for non parallel
        install of hdf5.
    """
    X1, X2, y = generate_abnet_batch(
        words, clusters, iteration, output_file,
        features_getter, stacked_features_getter)

    if synchronize != None:
        synchronize.acquire()
    output_file = h5py.File(output_file)
    output_file.create_group(str(iteration))
    output_file[str(iteration)].create_dataset('X1', data=X1)
    output_file[str(iteration)].create_dataset('X2', data=X2)
    output_file[str(iteration)].create_dataset('y', data=y)
    output_file.attrs[str(iteration)] = True
    if synchronize != None:
        synchronize.release()


def run(word_cluster_file, alignement_features, stacked_features, output_file):
    words, clusters = read_word_clusters_file(word_cluster_file)
    features_getter = FeaturesAPI_cached(alignement_features)
    stacked_features_getter = FeaturesAPI_cached(stacked_fbanks)
    for i in range(n_epochs):
        generate_and_save_abnet_batch(
            words, clusters, i, output_file,
            features_getter, stacked_features_getter,
            synchronize=None)


if __name__ == '__main__':
    # get_clusters()
    n_epochs = 500
    dset = 'train'
    WORD_CLUSTERS_FILE = 'zs_buckeye-plp'.format(dset)
    # WORD_CLUSTERS_FILE = 'english_{}_words.txt'.format(dset)
    words, clusters = read_word_clusters_file(WORD_CLUSTERS_FILE)
    mfccs = '../zerospeech2015/english_feats/htk/mfcc.h5f'
    stacked_fbanks = '../abnet_experiments/fb_mvn_stacked7.h5f'
    features_getter = FeaturesAPI_cached(mfccs)
    stacked_features_getter = FeaturesAPI_cached(stacked_fbanks)
    print('number of words considered: {}'.format(len(words)))
    for i in range(n_epochs):
        t0 = time()
        print("Starting iteration {}...".format(i))
        pairs_same, pairs_diff = sample_train_batch(words, clusters, hard=False)
        X1_same, X2_same = get_same_data(
            pairs_same, mfccs, stacked_fbanks, features_getter, stacked_features_getter)
        print("'same' word pairs sampled, now sampling "
              "'different' word pairs")
        X1_diff, X2_diff = get_diff_data(
            pairs_diff, stacked_fbanks, stacked_features_getter)
        y_same = np.ones(X1_same.shape[0], dtype=np.int8)
        y_diff = np.zeros(X1_diff.shape[0], dtype=np.int8)
        X1, X2, y = map(np.concatenate, [
            [X1_same, X1_diff], [X2_same, X2_diff], [y_same, y_diff]])
        print("Saving to disc")
        np.savez_compressed('data_aren/X1_{}_{}'.format(dset, i), X1)
        np.savez_compressed('data_aren/X2_{}_{}'.format(dset, i), X2)
        np.savez_compressed('data_aren/y_{}_{}'.format(dset, i), y)
        print("All done and saved, iteration took "
              "{} seconds".format(time()-t0))
