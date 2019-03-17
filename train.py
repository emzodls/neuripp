import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from utils import *
from random import shuffle

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
                val_data=(None,None),refresh_data=None,max_length=max_length,save_name='',store_best_acc = True):

    best_val_acc = 0
    best_val_epoch = 0
    for epoch in range(n_epochs):

        ## Ensures you get new data every refresh_data epochs
        if refresh_data and type(refresh_data) is int and epoch % refresh_data == 0:
            dataset,pos_remain,neg_remain = mix_samples(pos_data,neg_data,pos_frac,neg_frac)
        elif not refresh_data:
            dataset, pos_remain, neg_remain = mix_samples(pos_data, neg_data, pos_frac, neg_frac)
        else:
            raise Exception
        test_data = dataset[:int(val_frac*len(dataset))]
        val_data = dataset[int(val_frac*len(dataset)):]

        x_train,y_train = zip(*test_data)
        x_train = np.array([sequence_to_hot_vectors(seq,normalize_length=max_length) for seq in x_train])
        y_train = np.array(y_train)
        x_val,y_val = val_data

        if x_val and y_val and len(val_data[0]) == len(val_data[1]):
            x_val,y_val = val_data
            x_val = np.array([sequence_to_hot_vectors(seq, normalize_length=max_length) for seq in x_val])
            y_val = np.array(y_val)
        elif val_frac >= 0:
                x_val, y_val = zip(*val_data)
                x_val = np.array([sequence_to_hot_vectors(seq, normalize_length=max_length) for seq in x_val])
                y_val = np.array(y_val)

        model.fit(x_train,y_train)

        if x_val and y_val:
            loss, acc = model.evaluate(x_val,y_val)
            if best_val_acc <= acc:
                best_val_acc = acc
                best_val_epoch = epoch

    return model,best_val_acc,best_val_epoch
#
# for epoch in range(n_epochs):
#     ## Shuffle Training Set Every Epoch and Save 5% for validation
#     print('Starting Epoch {}'.format(epoch+1))
#     negativeTrainSet, negativeTestSet = split_set(negativeTrainSet|negativeTestSet, 0.50)
#     negativeLantiTrainSet, negativeLantiTestSet = split_set(negativeLantiTrainSet|negativeLantiTestSet, 0.50)
#     print('Total Number of positives: {}'.format(len(positives)))
#     print('Total Number of negatives: {}'.format(len(negativeTrainSet | negativeLantiTrainSet)))
#     masterSet = [(neg, 0) for neg in negativeTrainSet]
#     masterSet.extend((neg, 0) for neg in negativeLantiTrainSet)
#     masterSet.extend((pos, 1) for pos in positives)
#     shuffle(masterSet)
#     trainingSet = masterSet
#     print('Training with {} positives and {} negatives, for a total of {} exemplars'.format(len(positives),
#                                                                                             len(negativeTrainSet|negativeLantiTrainSet),
#                                                                                             len(trainingSet)))
#     with open(logfile_name, 'a') as outfile:
#         outfile.write('Starting Epoch {}\n'.format(epoch+1 ))
#         outfile.write('Training with {} positives and {} negatives, for a total of {} exemplars\n'.format(len(positives),
#                                                                                             len(negativeTrainSet|negativeLantiTrainSet),
#                                                                                             len(trainingSet)))
#     trainingIdx = math.floor(len(trainingSet) * frac_train)
#     numberTrained = len(trainingSet[:trainingIdx])
#     current_loss = 0
#     for idx,labeled_pair in enumerate(trainingSet[:trainingIdx]):
#         model.train()
#         category, line, category_tensor, line_tensor = prepareTensors(labeled_pair)
#         output, loss = train(category_tensor,line_tensor)
#         current_loss += loss
#
#         if idx % print_every == 0:
#             guess, guess_i = category_from_output(output)
#             correct = '✓' if guess == category else '✗ (%s)' % category
#             print('%d %d%% (%s) %.4f %s / %s %s' % (
#                 idx, idx / numberTrained * 100, timeSince(start), current_loss/(idx+1), line, guess, correct))
#             with open(logfile_name, 'a') as outfile:
#                 outfile.write('%d %d%% (%s) %.4f %s / %s %s\n' % (
#                     idx, idx / numberTrained * 100, timeSince(start), current_loss/(idx+1), line, guess, correct))
#
#     print('Epoch {} finished, Average Loss = {}'.format(epoch + 1,current_loss/numberTrained))
#     all_losses.append(current_loss/numberTrained)
#     print('Testing Model with Validation Data.')