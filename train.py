import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from utils import *
from random import shuffle
from models import create_model_lstm,create_model_conv_lstm,create_model_conv,create_model_conv_parallel,create_model_conv_parallel_lstm
import os

positives_path = '/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/positives_all.fa'
negatives_path = '/Users/emzodls/Dropbox/Lab/Warwick/RiPP_nnets/final_train_sets/negatives_all.fa'

positive_sequences = process_fasta(positives_path)
negative_sequences = process_fasta(negatives_path)

negative_pairs = [(seq,0) for seq in negative_sequences]
positive_pairs = [(seq,1) for seq in positive_sequences]

max_length = 120

def mix_samples(set_a,set_b,set_a_frac=0.5,set_b_frac=0.5):

    shuffle(set_a)
    shuffle(set_b)
    set_a_idx = int(len(set_a)*set_a_frac)
    set_b_idx = int(len(set_b)*set_b_frac)
    master_set = set_a[:set_a_idx]
    master_set.extend(set_b[:set_b_idx])
    shuffle(master_set)

    return master_set,set_a[set_a_idx:],set_b[set_b_idx:]

def train_model(model,n_epochs,pos_data,neg_data,pos_frac=0.5,neg_frac=0.5,val_frac=0,
                val_data=(None,None),refresh_data=None,max_length=max_length,save_name='',store_best_acc = True,
                wait_until=1000,logfile=None):

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
        elif val_frac >= 0:
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
    cnn = create_model_conv()
    cnn.summary()
    train_model(cnn, 150, positive_pairs, negative_pairs, pos_frac=1.0, neg_frac=0.5, val_frac=0.15,
                refresh_data=5, save_name='cnn_linear',wait_until=20,logfile='cnn_linear.log')

    cnn_parallel = create_model_conv_parallel()
    cnn_parallel.summary()
    train_model(cnn_parallel, 150, positive_pairs, negative_pairs, pos_frac=1.0, neg_frac=0.5, val_frac=0.15,
                refresh_data=5, save_name='cnn_parallel', wait_until=20, logfile='cnn_parallel.log')

    cnn_lstm = create_model_conv_lstm()
    cnn_lstm.summary()
    train_model(cnn_lstm, 150, positive_pairs, negative_pairs, pos_frac=1.0, neg_frac=0.5, val_frac=0.15,
                refresh_data=5, save_name='cnn_linear_lstm',wait_until=20,logfile='cnn_linear_lstm.log')

    cnn_lstm_parallel = create_model_conv_parallel_lstm()
    cnn_lstm_parallel.summary()
    train_model(cnn_lstm_parallel, 150, positive_pairs, negative_pairs, pos_frac=1.0, neg_frac=0.5, val_frac=0.15,
                refresh_data=5, save_name='cnn_linear_lstm',wait_until=20,logfile='cnn_parallel_lstm.log')

    lstm = create_model_lstm()
    lstm.summary()
    train_model(lstm, 150, positive_pairs, negative_pairs, pos_frac=1.0, neg_frac=0.5, val_frac=0.15,
                refresh_data=5, save_name='lstm_layer',wait_until=20,logfile='lstm.log')
