# Copyright (C) 2019 Emmanuel LC. de los Santos
# University of Warwick
# Warwick Integrative Synthetic Biology Centre
#
# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.
'''
    This file is part of NeuRiPP.

    NeuRiPP is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    NeuRiPP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with NeuRiPP.  If not, see <http://www.gnu.org/licenses/>.
'''

import tensorflow
from tensorflow import keras
from utils import *
from random import shuffle
from models import create_model_lstm,create_model_conv_lstm,\
    create_model_conv,create_model_conv_parallel,create_model_conv_parallel_lstm
import argparse
import logging
from glob import glob


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
                x_val=[],y_val=[],refresh_data=None,max_length=120,save_name='',store_best_acc = True,
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

        if not len(x_val) > 0  and not len(y_val) > 0 and val_frac > 0:
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
                        outfile.write('No Improvement Model: {}, ({} times)\n'.format(epoch + 1, no_improvement))
            if no_improvement >= wait_until:
                break


    return model,best_val_acc,best_val_epoch

def check_model_fasta(stored_model,fasta_file,label):
    x_test,y_test = prepare_input_vector(process_fasta(fasta_file),label)
    model = keras.models.load_model(stored_model)
    loss, acc = model.evaluate(x_test, y_test)
    return(loss,acc)

def check_model_tuple(stored_model,data):
    x_test,y_test = data
    model = keras.models.load_model(stored_model)
    loss, acc = model.evaluate(x_test, y_test)
    return(loss,acc)

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_frac(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError("{} must be between 0.0 and 1.0".format(value))
    return fvalue

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("-pos_frac",type=check_frac,help="Fraction of Positive Dataset to Use for Training",default=1.0)
    parser.add_argument("-neg_frac", type=check_frac, help="Header for Saved Model File", default=1.0)
    parser.add_argument("-e","--epochs",type=check_positive,default=100,help="Number of Epochs with Training Set")
    parser.add_argument("-r","--refresh_every",type=check_positive,default=5,help="Reshuffle Training Data Every n "
                                                                                  "Epochs (only works if fraction of Training Set is taken)")
    parser.add_argument('-m','--model',type=str,choices=['cnn-parallel','cnn-linear','cnn-linear-lstm','cnn-parallel-lstm','lstm'],default="cnn-parallel",
                        help="Specify Base Model to Train")
    parser.add_argument('-l','--max_len',type=check_positive,default=120,help="Assumed Maximum Length of Precursor Peptide (truncates amino acids after this length)")
    parser.add_argument('-w', '--wait_until', type=check_positive, default=100,
                        help="Number of Rounds allowed for no improvement in training set before terminating.")
    parser.add_argument("-outname", type=str, help="Header for Model Training Files", default="model")
    parser.add_argument("-outdir",type=str,help="Path to Output Directory",default=os.getcwd())
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-val_frac',type=check_frac,
                       help="Set aside fraction of positive and negative set to use for Validation", default=0)
    group.add_argument('-val_set', type=argparse.FileType('r'),nargs=2,help="Fasta Files Corresponding to Positive and Negative Test Sets")
    parser.add_argument("-pos", type=argparse.FileType('r'), help="Path to Fasta File Containing Positive Sequences.",
                        required=True)
    parser.add_argument("-neg", type=argparse.FileType('r'), help="Path to Fasta File Containing Negative Sequences.",
                        required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    else:
        model_files = glob(os.path.join(args.outdir,'{}*'.format(args.outname)))
        if len(model_files) > 0:
            print("Warning model files for model name, {}, exist. These will be overwritten".format(args.outname))


    positives_path = args.pos
    negatives_path = args.neg

    positive_sequences = process_fasta(positives_path)
    negative_sequences = process_fasta(negatives_path)

    negative_pairs = [(seq, 0) for seq in negative_sequences]
    positive_pairs = [(seq, 1) for seq in positive_sequences]

    max_length = 120
    shuffle(positive_pairs)
    shuffle(negative_pairs)

    # take 550 out (~20%) of the positive set, 1650 out of the negative set for validation (8.5%) , 2176 left in pos set
    pos_idx = int((1-args.val_frac)*len(positive_pairs))
    neg_idx = int((1-args.val_frac)*len(negative_pairs))
    print(pos_idx,neg_idx)

    train_pos = positive_pairs[:pos_idx]
    train_neg = negative_pairs[:neg_idx]

    test_pos = positive_pairs[pos_idx:]
    test_neg = negative_pairs[neg_idx:]

    with open(os.path.join(args.outdir,'{}_train_set.fa'.format(args.outname)),'w') as outfile:
        for i,(seq,pos) in enumerate(train_pos):
            outfile.write('>pos_{}\n{}\n'.format(i+1,seq.upper()))
        for i,(seq,neg) in enumerate(train_neg):
            outfile.write('>neg_{}\n{}\n'.format(i+1,seq.upper()))

    x_test = None
    y_test = None
    if args.val_set and len(args.val_set) == 2:
        positives_path = args.val_set[0]
        negatives_path = args.val_set[1]

        positive_sequences = process_fasta(positives_path)
        negative_sequences = process_fasta(negatives_path)

        with open(os.path.join(args.outdir,'{}_test_set.fa'.format(args.outname)),'w') as outfile:
            for i,seq in enumerate(positive_sequences):
                outfile.write('>pos_{}\n{}\n'.format(i+1,seq.upper()))
            for i,seq in enumerate(negative_sequences):
                outfile.write('>neg_{}\n{}\n'.format(i+1,seq.upper()))

        negative_pairs = [(seq, 0) for seq in negative_sequences]
        positive_pairs = [(seq, 1) for seq in positive_sequences]

        val_data = positive_pairs+negative_pairs
        x_test, y_test = zip(*val_data)
        x_test = np.array([sequence_to_hot_vectors(seq, normalize_length=max_length) for seq in x_test])
        y_test = np.array(y_test)
    elif args.val_frac > 0:
        with open(os.path.join(args.outdir,'{}_test_set.fa'.format(args.outname)),'w') as outfile:
            for i,(seq,pos) in enumerate(test_pos):
                outfile.write('>pos_{}\n{}\n'.format(i+1,seq.upper()))
            for i,(seq,neg) in enumerate(test_neg):
                outfile.write('>neg_{}\n{}\n'.format(i+1,seq.upper()))
        val_data = test_pos+test_neg
        x_test, y_test = zip(*val_data)
        x_test = np.array([sequence_to_hot_vectors(seq, normalize_length=max_length) for seq in x_test])
        y_test = np.array(y_test)
    else:
        print("No Validation Data, quitting")

    if x_test is not None and y_test is not None and (len(x_test) == len(y_test)):
        models = {'cnn-parallel': create_model_conv_parallel, 'cnn-linear': create_model_conv,
                  'cnn-linear-lstm': create_model_conv_lstm,
                  'cnn-parallel-lstm': create_model_conv_parallel_lstm, 'lstm': create_model_lstm}
        n_epochs = args.epochs
        model = models[args.model]()
        train_model(model, n_epochs, train_pos, train_neg, pos_frac=args.pos_frac, neg_frac=args.neg_frac,
                    refresh_data=args.refresh_every, save_name=os.path.join(args.outdir,args.outname),
                    wait_until=args.wait_until,logfile=os.path.join(args.outdir,'{}.log'.format(args.outname)),x_val=x_test,y_val=y_test)