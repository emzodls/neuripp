import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from utils import *
from random import shuffle
from models import create_model_lstm,create_model_conv_lstm,create_model_conv,create_model_conv_parallel,create_model_conv_parallel_lstm
import os


def mix_samples(set_a,set_b,set_a_frac=0.5,set_b_frac=0.5):

    shuffle(set_a)
    shuffle(set_b)
    set_a_idx = int(len(set_a)*set_a_frac)
    set_b_idx = int(len(set_b)*set_b_frac)
    master_set = set_a[:set_a_idx]
    master_set.extend(set_b[:set_b_idx])
    shuffle(master_set)

    return master_set,set_a[set_a_idx:],set_b[set_b_idx:]

def train_model(model,n_epochs,pos_data,neg_data,pos_frac=0.5,neg_frac=0.5,val_frac=-0.1,
                val_data=(None,None),refresh_data=None,max_length=120,save_name='',store_best_acc = True,
                wait_until=1000,logfile=None,reset_weights=False):

    if logfile:
        with open(logfile,'a') as outfile:
            outfile.write('Testing {}\n'.format(save_name))
    best_val_acc = 0
    best_val_epoch = 0
    no_improvement = 0
    dataset, pos_remain, neg_remain = mix_samples(pos_data, neg_data, pos_frac, neg_frac)
    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch+1))
        ## Ensures you get new data every refresh_data epochs
        if refresh_data and type(refresh_data) is int and epoch % refresh_data == 0:
            dataset,pos_remain,neg_remain = mix_samples(pos_data,neg_data,pos_frac,neg_frac)
        test_data = dataset[:int((1-val_frac)*len(dataset))]
        val_frac_data = dataset[int((1-val_frac)*len(dataset)):]
        shuffle(test_data)
        x_train,y_train = zip(*test_data)
        x_train = np.array([sequence_to_hot_vectors(seq,normalize_length=max_length) for seq in x_train])
        y_train = np.array(y_train)

        if val_data[0] and val_data[1] and len(val_data[0]) == len(val_data[1]):
            x_val,y_val = val_data
            x_val = np.array([sequence_to_hot_vectors(seq, normalize_length=max_length) for seq in x_val])
            y_val = np.array(y_val)
        elif val_frac > 0:
            x_val, y_val = zip(*val_frac_data)
            x_val = np.array([sequence_to_hot_vectors(seq, normalize_length=max_length) for seq in x_val])
            y_val = np.array(y_val)

        if len(x_val) != 0 and len(x_val) == len(y_val):


            output = model.fit(x_train, y_train, batch_size=5)

            loss, acc = model.evaluate(x_val,y_val)
            if logfile:
                with open(logfile, 'a') as outfile:
                    outfile.write('Epoch {}:, Acc: {:.04f}, Loss: {:.04f}\n'.format(epoch+1, output.history['accuracy'][0]
                                                                                ,output.history['loss'][0]))
            if best_val_acc < acc:
                best_val_acc = acc
                best_val_epoch = epoch + 1
                print('Saving Model: {}, Acc: {}'.format(best_val_epoch,best_val_acc))
                if logfile:
                    with open(logfile, 'a') as outfile:
                        outfile.write('Saving Model: {}, Acc: {}\n'.format(best_val_epoch,best_val_acc))
                keras.models.save_model(model,'{}-epoch_{}-acc_{:.04f}.hdf5'.format(save_name,
                                                                                    best_val_epoch,best_val_acc))
                model.save_weights(save_name)
                no_improvement = 0
            else:
                if reset_weights:
                    model.load_weights(save_name)
                no_improvement += 1
                print('No Improvement Model: {}, ({} times)'.format(epoch + 1, no_improvement))
                if logfile:
                    with open(logfile, 'a') as outfile:
                        outfile.write('No Improvement Model: {}, ({} times)'.format(epoch + 1, no_improvement))
            if no_improvement >= wait_until:
                break


    return model,best_val_acc,best_val_epoch


if __name__ == '__main__':

    positives_path = 'positives_all.fa'
    negatives_path = 'negatives_all.fa'

    positive_sequences = process_fasta(positives_path)
    negative_sequences = process_fasta(negatives_path)

    negative_pairs = [(seq, 0) for seq in negative_sequences]
    positive_pairs = [(seq, 1) for seq in positive_sequences]

    max_length = 120
    shuffle(positive_pairs)
    shuffle(negative_pairs)

    # take 550 out (~20%) of the positive set, 1650 out of the negative set for validation (8.5%) , 2176 left in pos set

    test_pos = positive_pairs[:550]
    test_neg = negative_pairs[:1650]

    train_pos = positive_pairs[550:]
    train_neg = negative_pairs[1650:]

    with open('train_set.fa','w') as outfile:
        for i,(seq,pos) in enumerate(test_pos):
            outfile.write('>pos_{}\n{}\n'.format(i+1,seq.upper()))
        for i,(seq,neg) in enumerate(test_neg):
            outfile.write('>neg_{}\n{}\n'.format(i+1,seq.upper()))

    cnn = create_model_conv()
    cnn.summary()
    train_model(cnn, 200, train_pos, train_neg, pos_frac=1.0, neg_frac=0.35,
                refresh_data=5, save_name='cnn_linear',wait_until=50,logfile='cnn_linear.log',val_data=(test_pos,test_neg))

    cnn_parallel = create_model_conv_parallel()
    cnn_parallel.summary()
    train_model(cnn_parallel, 200, train_pos, train_neg, pos_frac=1.0, neg_frac=0.35,
                refresh_data=5, save_name='cnn_parallel', wait_until=50, logfile='cnn_parallel.log',val_data=(test_pos,test_neg))

    cnn_lstm = create_model_conv_lstm()
    cnn_lstm.summary()
    train_model(cnn_lstm, 200, train_pos, train_neg, pos_frac=1.0, neg_frac=0.35,
                refresh_data=5, save_name='cnn_linear_lstm',wait_until=50,logfile='cnn_linear_lstm.log',val_data=(test_pos,test_neg))

    cnn_lstm_parallel = create_model_conv_parallel_lstm()
    cnn_lstm_parallel.summary()
    train_model(cnn_lstm_parallel, 200, train_pos, train_neg, pos_frac=1.0, neg_frac=0.35,
                refresh_data=5, save_name='cnn_parallel_lstm',wait_until=50,logfile='cnn_parallel_lstm.log',val_data=(test_pos,test_neg))

    lstm = create_model_lstm()
    lstm.summary()
    train_model(lstm, 200, train_pos, train_neg, pos_frac=1.0, neg_frac=0.35,
                refresh_data=5, save_name='lstm_layer',wait_until=50,logfile='lstm.log',val_data=(test_pos,test_neg))
