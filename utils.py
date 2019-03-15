import numpy as np
import tensorflow as tf
from tensorflow import keras

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

