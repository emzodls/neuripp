import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from tensorflow.keras.layers import LSTM, Lambda
from tensorflow.keras.layers import TimeDistributed, Bidirectional
import numpy as np
import tensorflow as tf
import re
import tensorflow.keras.callbacks
import sys
import os
from random import shuffle


class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))

if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])

sequences = []
labels = []
pos_ctr = 0
for line in open('/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/positives_all.fa'):
    if not line.startswith('>'):
        sequences.append(line.strip().lower())
        pos_ctr+= 1
        labels.append(1)
neg_seqs = []
neg_labels = []
for line in open('/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/negatives_all.fa'):
    if not line.startswith('>'):
        neg_seqs.append(line.strip().lower())
        neg_labels.append(0)

shuffle(neg_seqs)
sequences.extend(neg_seqs[:int(pos_ctr*3)])
labels.extend(neg_labels[:int(pos_ctr*3)])
labeled_seqs = list(zip(sequences,labels))
shuffle(labeled_seqs)
sequences,labels = zip(*labeled_seqs)
aa_indices = dict((a, i+1) for i, a in enumerate('acdefghiklmnpqrstvwy'))


max_length = 120
X = np.ones((len(sequences),120), dtype=np.int64) * - 1
y = np.array(labels)

filter_length = [5, 3, 3]
nb_filter = [75, 75, 150]
pool_length = 2

seq_matrix = np.zeros((len(sequences),120,20),dtype=np.int64)
for i,sequence in enumerate(sequences):
    for j,aa in enumerate(sequence):
        seq_matrix[(i,120-1-j,aa_indices[aa]-1)] = 1


# labeled_seqs = list(zip(seq_matrix,y))
# shuffle(labeled_seqs)
# seq_matrix,y = zip(*labeled_seqs)

in_sequence =  Input(shape=(120,20))
embedded = Conv1D(filters=nb_filter[0],
                  kernel_size=filter_length[0],
                  padding='valid',
                  activation='relu',
                  kernel_initializer='glorot_normal',
                  strides=1)(in_sequence)
embedded = Dropout(0.1)(embedded)
embedded = MaxPooling1D(pool_size=pool_length)(embedded)
for i in range(1,len(nb_filter)):
    embedded = Conv1D(filters=nb_filter[i],
                      kernel_size=filter_length[i],
                      padding='valid',
                      activation='relu',
                      kernel_initializer='glorot_normal',
                      strides=1)(embedded)
    embedded = Dropout(0.1)(embedded)
    embedded = MaxPooling1D(pool_size=pool_length)(embedded)

bi_lstm_seq = \
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(embedded)

label = Dropout(0.3)(bi_lstm_seq)
label = Dense(64, activation='relu')(label)
label = Dense(2, activation='sigmoid')(label)
# sentence encoder
labeler = Model(inputs=in_sequence, outputs=label)
labeler.summary()

positives = []
pos_labels = []
for line in open('/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/positives_all.fa'):
    if not line.startswith('>'):
        positives.append(line.strip().lower())
        pos_labels.append(1)
negatives = []
neg_labels = []
for line in open('/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/negatives_all.fa'):
    if not line.startswith('>'):
        negatives.append(line.strip().lower())
        neg_labels.append(0)

test_set = positives + neg_seqs[int(pos_ctr*3):int(pos_ctr*5)]
test_labels = np.array([1 for x in positives] + [0 for x in neg_seqs[int(pos_ctr*3):int(pos_ctr*5)]])
test_X = np.zeros((len(test_set),120,20), dtype=np.int64)
for i,sequence in enumerate(test_set):
    for j,aa in enumerate(sequence):
        test_X[(i,120-1-j,aa_indices[aa]-1)] = 1

file_name = os.path.basename(sys.argv[0]).split('.')[0]
check_cb = tf.keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.{epoch:02d}-{val_accuracy:.2f}.hdf5',
                                           monitor='val_accuracy',
                                           verbose=1,  mode='max',save_best_only=True)
earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, mode='auto',restore_best_weights=True)
history = LossHistory()
labeler.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

labeler.fit(seq_matrix,labels,validation_data = (test_X ,test_labels),
          epochs=50, shuffle=True, callbacks=[check_cb, history,earlystop_cb])

positives = []
pos_labels = []
for line in open('/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/positives_all.fa'):
    if not line.startswith('>'):
        positives.append(line.strip().lower())
        pos_labels.append(1)
negatives = []
neg_labels = []
for line in open('/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/negatives_all.fa'):
    if not line.startswith('>'):
        negatives.append(line.strip().lower())
        neg_labels.append(0)

pos_X = np.zeros((len(positives),120,20), dtype=np.int64)
pos_y = np.array(pos_labels)

neg_X = np.zeros((len(negatives),120,20), dtype=np.int64)
neg_y = np.array(neg_labels)

for i,sequence in enumerate(positives):
    for j,aa in enumerate(sequence):
        pos_X[(i,120-1-j,aa_indices[aa]-1)] = 1

for i,sequence in enumerate(negatives):
    for j,aa in enumerate(sequence):
        neg_X[(i,120-1-j,aa_indices[aa]-1)] = 1

labeler.evaluate(pos_X,pos_labels)
labeler.evaluate(neg_X,neg_labels)
tf.keras.models.save_model(labeler,'labeler.hdf5')
