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
from utils import classify_peptides
from models import create_model_lstm,create_model_conv_lstm,\
    create_model_conv,create_model_conv_parallel,create_model_conv_parallel_lstm
import argparse
from glob import glob
import os

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m','--model',type=str,choices=['cnn-parallel','cnn-linear','cnn-linear-lstm','cnn-parallel-lstm','lstm'],default="cnn-parallel",
                        help="Model Architecture",required=True)
    parser.add_argument('-w','--weights',type=str, default=None,
                        help="Weights File for Model")
    parser.add_argument("-outname", type=str, help="Prefix for Positive or Negative Data", default="peptide")
    parser.add_argument("-outdir",type=str,help="Path to Output Directory",default=os.getcwd())
    parser.add_argument('--keep_negatives',action='store_true')
    parser.add_argument("-b",'--batch_size',type=check_positive,default=1000,help="Number of Samples to Give Model at a Time")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("input", type=argparse.FileType('r'), help="Fasta File Containing Sequences to Classify",
                        required=True)

    args = parser.parse_args()

    models = {'cnn-parallel': create_model_conv_parallel, 'cnn-linear': create_model_conv,
              'cnn-linear-lstm': create_model_conv_lstm,
                  'cnn-parallel-lstm': create_model_conv_parallel_lstm, 'lstm': create_model_lstm}
    model = models[args.model]()
    if os.path.isfile(args.weights):
        model.load_weights(args.weights)
        print("Successfully Loaded Weights for Model")
    classify_peptides(model, args.input, batch_size=args.batch_size, max_len=120,
                      output_name=args.outname, output_dictionary=args.outdir, output_negs=args.keep_negatives)