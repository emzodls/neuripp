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
    for line in open(path):
        if not line.startswith('>'):
            if line[-1] == '*':
                line = line[:-1]
            line = line.strip().lower()
            if len(set(line)-allowed_aas) == 0:
                sequences.append(line)
    return sequences

def classify_peptides(model_path,fasta_file,batch_size=1000,output_file=None,output_dictionary=False):
    if os.path.isfile(output_file):
        os.remove(output_file)
    fasta_dict = {}
    classification = {}
    allowed_aas = set('acdefghiklmnpqrstvwy')
    model = keras.models.load_model(model_path)
    fasta_entries = SeqIO.parse(fasta_file,'fasta')
    for idx,entry in enumerate(fasta_entries):
        seq = str(entry.seq).lower()
        if seq[-1] == '*':
            seq = seq[:-1]
        if len(set(seq) - allowed_aas) == 0:
            fasta_dict[entry.id] = str(seq)
        if (idx + 1) % batch_size == 0:
            order = sorted(list(fasta_dict.keys()))
            test_x = np.array([sequence_to_hot_vectors(fasta_dict[seq],normalize_length=120) for seq in order])
            guesses = model.predict(test_x)
            id = [np.argmax(x) for x in guesses]
            guess_dict = dict(zip(order,id))
            if output_file:
                with open(output_file,'a') as outfile:
                    for fasta_tag,guess in guess_dict.items():
                        if guess == 1:
                            outfile.write('>{}\n{}\n'.format(fasta_tag,fasta_dict[fasta_tag].upper()))
            if output_dictionary:
                classification.update(guess_dict)
            fasta_dict = {}
    else:
        order = sorted(list(fasta_dict.keys()))
        test_x = np.array([sequence_to_hot_vectors(fasta_dict[seq], normalize_length=120) for seq in order])
        guesses = model.predict(test_x)
        id = [np.argmax(x) for x in guesses]
        guess_dict = dict(zip(order, id))
        if output_file:
            with open(output_file, 'a') as outfile:
                for fasta_tag, guess in guess_dict.items():
                    if guess == 1:
                        outfile.write('>{}\n{}\n'.format(fasta_tag, fasta_dict[fasta_tag].upper()))
        if output_dictionary:
            classification.update(guess_dict)
    if output_dictionary:
        return classification
    else:
        return None