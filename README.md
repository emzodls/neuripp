# NeuRiPP (**Neu**ral Network Identification of **Ri**PP **P**recursor **P**eptides)
NeuRiPP is a neural network framework designed for classifying peptide sequences as putative precursor peptide sequences for RiPP biosynthetic gene clusters. It consists of two modules:
1. classify - Given a set of model weights and a fasta file can classify sequences as putative RiPP sequences. 
2. train - Given a set of positive and negative sequences as fasta files, train a specific neural network architecture to optimize its weights.

A description of the different neural network architectures and NeuRiPP's performance can be found in the bioRxiV.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

NeuRiPP requires Python 3, [Tensorflow 2.0 Alpha](https://www.tensorflow.org/install/), and [Biopython](https://biopython.org/). It has been tested on Python 3.6 and Python 3.7. 


### Installing

The following are general instructions to install NeuRiPP in OS X or Linux, NeuRiPP should be compatible in windows if Python 3, Tensorflow 2.0 alpha, and Biopython are installed.

(optional) It is recommended that you set up a separate virtual environment for NeuRiPP. You can do this in your shell using the command:
```
python3 -m venv /path/to/venv/neuripp
```
You can then activate the virtual environment using:
```
source /path/to/venv/neuripp/bin/activate
```
1. Clone the repository into your preferred path:
```
git clone https://github.com/emzodls/neuripp.git /path/to/neuripp
```
2. Install the prequisites. If you have pip or a package manager you can do this by using the command:
```
pip (or pip3) install -r requirements.txt
```
3. Once the prequisites are installed you can begin using NeuRiPP
## Using NeuRiPP

### Classifying Peptides
NeuRiPP takes a fasta file of peptides sequences as its input. It only recognizes sequences that have the standard 20 amino acid one letter code (i.e. sequences that have "X" or other non-standard amino acids **will be ignored**). In order for the classification to be accurate, weights for the model specified must be loaded as the neural network weights are initially randomized. Pretrained weights for each model described in the publication can be found in the weights folder. Alternatively weights can be trained using the train module.

You can classify peptides in a fasta file using the command:
```
python classify.py -w /path/to/weights -i <input.fasta>
```
This will write a file in the directory containing NeuRiPP called "peptide_pos.fa" which contains the sequences that NeuRiPP classifies as putative precursor peptides. By default NeuRiPP will use the cnn-parallel neural network archictecture.

You can change the architecture NeuRiPP uses using the "-m" flag and setting the model architecture to any of the five models described in the publication: cnn-parallel,cnn-linear,cnn-linear-lstm,cnn-parallel-lstm,lstm. **It is important that you use the appropriate weights for the selected model.** 
```
python classify.py -m cnn-linear -w /cnn/linear/weights -i <input.fasta>
```

You have the option of keeping negative sequences using the flag "--keep_negatives" if this flag is activated another file "peptides_neg.fa" will be generated which contains the sequences in the input file that NeuRiPP thinks are not precursor peptides.

Finally, you can change the prefix from peptides to a different name using the flag "-outname" likewise, a different output directory can be specified using the flag "-outdir"

**Sample Usage**
```
python classify.py -m cnn-parallel-lstm -w weights -outdir /ripp_precursors/ -outname ripps_cnn_parallel_lstm --keep_negatives -i <input.fasta>
```
This command write "ripps_cnn_parallel_lstm_pos.fa" and "ripps_cnn_parallel_lstm_neg.fa" into the /ripp_precursors directory
### Training New Weights for Model Architectures

NeuRiPP allows you to train your own weights for the different model architectures using two fasta files containing positive examples, and negative examples as inputs.

To train a model a set of sequences used to test the weight optimization after each round **is required**. This can either be a specified fraction of the positive and negative sets using the flag -val_frac, or a separate set of fasta files using the flag -val_set.
```
python train.py -val_set pos_test.fa neg_test.fa -pos positives.fa -neg negatives.fa 
```
This command will generate a file "model.hdf5" which will be weights trained for the cnn-parallel model, using positives.fa and negatives.fa to optimize the weights, and the files pos_test.fa  and neg_test.fa as the validation set after every round of training. 

If a fraction of the training sets is excluded for training and testing, the test and training sets can be resampled using the -r/--refresh_every flag:
```
python train.py -pos_frac 0.8 -neg_frac 0.4 -r 10 -val_frac 0.15 -pos positives.fa -neg negatives.fa

```
The model architecture for training weights can be specified using the -m flag.

As with the classify module, the output directory and output name can be changed from the default by using the flags -outname and -outdir

Model Weights were optimized in the paper using commands such as:
```
python train.py -m lstm -pos_frac 1.0 -neg_frac 0.4 -r 5 -val_frac 0.15 -e 200 -w 50 -outname lstm -outdir optimized_weights -pos pos_all.fa -neg neg_all.fa
```

## Authors

* **Emzo de los Santos** - https://warwick.ac.uk/fac/sci/lifesci/people/edelossantos/

## License

This project is licensed under the GNU AGPL v3
