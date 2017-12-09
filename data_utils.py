from config import *
from scipy import misc
import numpy as np

class ImageGenerator():
    def __init__(self, sequence, batch_size):
        # file name, x, y, z
        self.sequence = np.array(sequence)
        self.batch_size = batch_size

    def next(self):
        num_examples = len(self.sequence)
        perm_ind = np.random.permutation(num_examples)
        while True:
            for i in range(num_examples, self.batch_size):
                batch_names = self.sequence[perm_ind[i:i + self.batch_size]]
                X_batch = np.array([misc.imread(filename[0]) / 255.0 for filename in batch_names])
                Y_batch = np.array([filename[1] / 255.0 for filename in batch_names])
                yield X_batch, Y_batch


class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size):
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        chunk_size = int(1 + (len(sequence) - 1) / batch_size)
        self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]

    def next(self):
        while True:
            output = []
            for i in range(self.batch_size):
                idx = self.indices[i]
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
                assert len(left_pad) == LEFT_CONTEXT
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len
                self.indices[i] = (idx + self.seq_len) % len(self.sequence)
                images, targets = zip(*result)
                images_left_pad, _ = zip(*left_pad)
                output.append((np.stack(images_left_pad + images), np.stack(targets)))
            output = list(zip(*output))
            output[0] = np.stack(output[0]) # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1]) # batch_size x seq_len x OUTPUT_DIM
            return output

def read_csv(filename, train=True):
    with open(filename, 'r') as f:
        if train:
            lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()][1:]
        else:
            lines = [ln.strip().split(",") for ln in f.readlines()][1:]
        prefix = './data/train/output/' if train else './data/test/center/'
        ext = '' if train else '.jpg'
        lines = map(lambda x: (prefix + x[0] + ext, np.float32(x[1:])), lines) # imagefile, outputs
        return list(lines)

def process_csv(filename, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    train_seq = []
    valid_seq = []
    cnt = 0
    for ln in lines:
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val):
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    print (len(train_seq), len(valid_seq))
    print (mean, std) # we will need these statistics to normalize the outputs (and ground truth inputs)
    return ((train_seq, valid_seq), (mean, std))
