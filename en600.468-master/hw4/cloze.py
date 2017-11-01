from __future__ import print_function
import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable

import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Predictor for Word Cloze.")
parser.add_argument("--data_file", required=True,
                    help="File for data set.")
parser.add_argument("--model_file", required=True,
                    help="Location to load the models.")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def embedding_to_word(embedding, vocab):
    return ' '.join(vocab.itos[i] for i in embedding.data)

def main(options):

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    train, dev, test, vocab = torch.load(open(options.data_file, 'rb'), pickle_module=dill)

    batched_test, batched_test_mask = utils.tensor.advanced_batchize_no_sort(test, 1, vocab.stoi["<pad>"])
    #batched_test, batched_test_mask, _ = utils.tensor.advanced_batchize(test, 1, vocab.stoi["<pad>"])

    vocab_size = len(vocab)

    rnnlm = torch.load(options.model_file)
    if use_cuda > 0:
        rnnlm.cuda()
    else:
        rnnlm.cpu()

    rnnlm.eval()
    m = 0
    for line in test:
        print(m, file=sys.stderr)
        m += 1
        blanks = []
        for i in range(len(line)):
            if vocab.itos[line[i]] == '<blank>':
                blanks.append(i)
        # print(blanks)
        # print(line)
        test_in = Variable(line).unsqueeze(1)
        test_out = rnnlm(test_in)
        test_out = test_out.view(-1, vocab_size)
        cur = []
        newCur = []
        for i in blanks:
            # if vocab.itos[line[i]] == '<blank>':
                # print(test_out[i][1:])
            _, argmax = torch.max(test_out[i][1:], 0)
            newCur.append(argmax.data[0] + 1)
        count = 0
        while newCur != cur and count < 20:
            # print(7)
            count += 1
            # print(cur, newCur)
            cur = newCur
            newCur = []
            # print(cur)
            for j, i in enumerate(blanks):
                # print(j, i, blanks)
                line[i] = cur[j]
            test_in = Variable(line).unsqueeze(1)
            test_out = rnnlm(test_in)
            test_out = test_out.view(-1, vocab_size)
            for i in blanks:
                print ("b")
                print (test_out[i])
                print ("a")
                # if vocab.itos[line[i]] == '<blank>':
                    # print(test_out[i][1:])
                _, argmax = torch.max(test_out[i][1:], 0)
                newCur.append(argmax.data[0] + 1)
            # print(line)
        cur = []
        for val in newCur:
            cur.append(vocab.itos[val])
        print(' '.join(cur).encode('utf-8').strip())
        

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)