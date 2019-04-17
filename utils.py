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

import numpy as np
from Bio import SeqIO
import os
from models import *
from glob import glob

def sequence_to_idx(sequence,normalize_length=None):
    '''
    :param sequence: amino acid sequence
    :return: list of indices for embedding
    given an amino acid sequence will return a mapping of to integers (0 is reserved for padding)
    '''
    if normalize_length:
        sequence = sequence[:normalize_length]
    sequence = sequence.lower()

    aa_indices = dict((a, i + 1) for i, a in enumerate('acdefghiklmnpqrstvwy'))
    assert len(set(sequence) - set(aa_indices.keys())) == 0
    if normalize_length:
        output = np.zeros(normalize_length)
    else:
        output = np.zeros(len(sequence))
    for i,aa in enumerate(sequence):
        output[i] = aa_indices[aa]
    return(output)

def sequence_to_hot_vectors(sequence,normalize_length=None):
    '''
    :param sequence: amino acid sequence
    :return: len(seq) x 20 matrix with 1 corresponding to the index of the amino acid
    '''
    if normalize_length:
        sequence = sequence[:normalize_length]
    sequence = sequence.lower()
    if normalize_length:
        seq_matrix = np.zeros((normalize_length, 20))
    else:
        seq_matrix = np.zeros((len(sequence), 20))
    indices = sequence_to_idx(sequence)
    for i,aa in enumerate(indices):
        seq_matrix[(i,int(aa)-1)] = 1
    return seq_matrix

def process_fasta(path):
    sequences = []
    allowed_aas = set('acdefghiklmnpqrstvwy')
    if type(path) is str:
        for line in open(path):
            if not line.startswith('>'):
                if line[-1] == '*':
                    line = line[:-1]
                line = line.strip().lower()
                if len(set(line)-allowed_aas) == 0:
                    sequences.append(line)
    else:
        for line in path:
            if not line.startswith('>'):
                if line[-1] == '*':
                    line = line[:-1]
                line = line.strip().lower()
                if len(set(line)-allowed_aas) == 0:
                    sequences.append(line)
    return sequences

def prepare_input_vector(sequences,label,max_len=120):
    '''
    :param sequences: iterable of peptide sequences
    :param label: 0 or 1
    :return: tuple of np arrays that can be fed to a model for evaluation or training
    '''
    x = np.array([sequence_to_hot_vectors(seq,normalize_length=max_len) for seq in sequences])
    y = np.array([label for seq in sequences])

    return(x,y)


def check_model_fasta(model_type,fasta_file,label,weight_file=None):
    models = {'cnn-parallel': create_model_conv_parallel, 'cnn-linear': create_model_conv,
              'cnn-linear-lstm': create_model_conv_lstm,
                  'cnn-parallel-lstm': create_model_conv_parallel_lstm, 'lstm': create_model_lstm}
    model = models[model_type]()
    if weight_file and os.path.isfile(weight_file):
        model.load_weights(weight_file)
        print("Successfully Loaded Weights for Model")
    x_test,y_test = prepare_input_vector(process_fasta(fasta_file),label)
    loss, acc = model.evaluate(x_test, y_test)
    return(loss,acc)

def classify_peptides(model,fasta_file,batch_size=1000,max_len=120,
                      output_name=None,output_dictionary=False,output_negs=False):

    fasta_dict = {}
    classification = {}
    allowed_aas = set('acdefghiklmnpqrstvwy')
    fasta_entries = SeqIO.parse(fasta_file,'fasta')
    if output_name:
        if os.path.isfile(output_name + '_pos.fa'):
            os.remove(output_name + '_pos.fa')
        if os.path.isfile(output_name + '_neg.fa'):
            os.remove(output_name + '_neg.fa')
    for idx,entry in enumerate(fasta_entries):
        seq = str(entry.seq).lower()
        if seq[-1] == '*':
            seq = seq[:-1]
        if len(set(seq) - allowed_aas) == 0:
            fasta_dict[entry.id] = str(seq)
        if (idx + 1) % batch_size == 0:
            order = sorted(list(fasta_dict.keys()))
            test_x = np.array([sequence_to_hot_vectors(fasta_dict[seq],normalize_length=max_len) for seq in order])
            guesses = model.predict(test_x)
            ids = [np.argmax(x) for x in guesses]
            scores = [np.log(x[np.argmax(x)] / x[np.argmin(x)]) for x in guesses]
            score_dict = dict(zip(order, scores))
            guess_dict = dict(zip(order, ids))
            if output_name:
                with open(output_name+"_pos.fa",'a') as outfile_pos:
                    for fasta_tag,guess in guess_dict.items():
                        if guess == 1:
                            outfile_pos.write('>{}|score:{:.2f}\n{}\n'.format(fasta_tag,score_dict[fasta_tag],fasta_dict[fasta_tag].upper()))
                        elif output_negs:
                            with open(output_name + "_neg.fa", 'a') as outfile_neg:
                                outfile_neg.write('>{}|score:{:.2f}\n{}\n'.format(fasta_tag, score_dict[fasta_tag],
                                                                              fasta_dict[fasta_tag].upper()))
            if output_dictionary:
                classification.update(guess_dict)
            fasta_dict = {}
    else:
        order = sorted(list(fasta_dict.keys()))
        test_x = np.array([sequence_to_hot_vectors(fasta_dict[seq], normalize_length= max_len) for seq in order])
        guesses = model.predict(test_x)
        ids = [np.argmax(x) for x in guesses]
        scores = [np.log(x[np.argmax(x)] / x[np.argmin(x)]) for x in guesses]
        score_dict = dict(zip(order, scores))
        guess_dict = dict(zip(order, ids))
        if output_name:
            with open(output_name + "_pos.fa", 'a') as outfile_pos:
                for fasta_tag, guess in guess_dict.items():
                    if guess == 1:
                        outfile_pos.write('>{}|score:{:.2f}\n{}\n'.format(fasta_tag, score_dict[fasta_tag],
                                                                          fasta_dict[fasta_tag].upper()))
                    elif output_negs:
                        with open(output_name + "_neg.fa", 'a') as outfile_neg:
                            outfile_neg.write('>{}|score:{:.2f}\n{}\n'.format(fasta_tag, score_dict[fasta_tag],
                                                                          fasta_dict[fasta_tag].upper()))
        if output_dictionary:
            classification.update(guess_dict)
    if output_dictionary:
        return classification
    else:
        return None