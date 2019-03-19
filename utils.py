import numpy as np
import tensorflow as tf
from tensorflow import keras
from Bio import SeqIO

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
            line = line.strip().lower()
            if len(set(line)-allowed_aas) == 0:
                sequences.append(line)
    return sequences

def classify_peptides(model_path,fasta_file,batch_size=1000,output_file=None):
    fasta_dict = {}
    classification = {}
    allowed_aas = set('acdefghiklmnpqrstvwy')
    model = keras.models.load_model(model_path)
    fasta_entries = SeqIO.parse(fasta_file,'fasta')
    for idx,entry in enumerate(fasta_entries):
        seq = str(entry.seq).lower()
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
                            outfile.write('>{}\n{}\n'.format(fasta_tag,fasta_dict[fasta_tag]))
            classification.update(guess_dict)
            fasta_dict = {}
    else:
        order = sorted(list(fasta_dict.keys()))
        test_x = np.array([sequence_to_hot_vectors(fasta_dict[seq], normalize_length=120) for seq in order])
        guesses = model.predict(test_x)
        print(guesses)
        id = [np.argmax(x) for x in guesses]
        guess_dict = dict(zip(order, id))
        if output_file:
            with open(output_file, 'a') as outfile:
                for fasta_tag, guess in guess_dict.items():
                    if guess == 1:
                        outfile.write('>{}\n{}\n'.format(fasta_tag, fasta_dict[fasta_tag]))
        classification.update(guess_dict)
    return classification
