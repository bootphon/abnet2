import glob
import os
import abnet2.generate_features as generate_features
import abnet2.train_and_eval as train_and_eval


#########
# Paths:
#########

wav_folder = ''
train_words = ''
val_words = ''

#########
# Generate features:
#########

files = glob.glob(os.path.join(wav_folder), '*.wav')
generate_features.generate_all(files, 'mfcc.h5f', 'fb_mvn_stacked.h5f')

#########
# Training and evaluating:
#########

train_and_eval.run(train_words, val_words, 'mfcc.h5f', 'fb_mvn_stacked.h5f')
