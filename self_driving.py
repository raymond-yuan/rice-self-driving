import tensorflow as tf
import numpy as np
import os
from config import *

from models import komada

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
        lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()][1:]
        prefix = './data/train/' if train else './data/test/center/'
        ext = '' if train else '.jpg'
        lines = map(lambda x: ('./data/imgs/' + x[0] + ext, np.float32(x[1:])), lines) # imagefile, outputs
        return lines

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


# In[38]:

(train_seq, valid_seq), (mean, std) = process_csv(filename="./data/imgs/interpolated.csv", val=5) # concatenated interpolated.csv from rosbags
test_seq = read_csv("challenge_2/exampleSubmissionInterpolatedFinal.csv", train=False) # interpolated.csv for testset filled with dummy values

# # Model
graph = tf.Graph()

with graph.as_default():
    # Build model
    model_out = komada.komada_model(mean=mean, std=std)

    learning_rate = model_out['lr']
    keep_prob = model_out['keep_prob']
    aux_cost_weight = model_out['aux_cost_wt']

    inputs = model_out['inputs']
    targets = model_out['targets']
    optimizer = model_out['train_step']
    steering_predictions = model_out['preds']

    controller_initial_state_autoregressive = model_out['ctrl_init_autoregressive']
    controller_final_state_gt = model_out['cntrl_final_gt']
    controller_final_state_autoregressive = model_out['ctrl_final_autoregressive']
    mse_autoregressive_steering = model_out['mse_autoreg_steering']

    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('v3/train_summary', graph=graph)
    valid_writer = tf.summary.FileWriter('v3/valid_summary', graph=graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)



# # Training
#
# At this point we can start the training procedure.
#
# We will perform optimization for 100 epochs, doing validation after each epoch. We will keep the model's version that obtains the best performance in terms of the primary loss (autoregressive steering MSE) on the validation set.
# An aggressive regularization is used (`keep_prob=0.25` for dropout), and the validation loss is highly non-monotonical.
#
# For each version of the model that beats the previous best validation score we will overwrite the checkpoint file and obtain predictions for the challenge test set.

# In[41]:

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

checkpoint_dir = os.getcwd() + "/v3"

global_train_step = 0
global_valid_step = 0

KEEP_PROB_TRAIN = 0.25

def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step
    test_predictions = {}
    valid_predictions = {}
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    total_num_steps = int(1 + (batch_generator.indices[1] - 1) / SEQ_LEN)
    controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
    acc_loss = np.float128(0.0)
    for step in range(total_num_steps):
        feed_inputs, feed_targets = batch_generator.next()
        feed_dict = {inputs : feed_inputs, targets : feed_targets}
        if controller_final_state_autoregressive_cur is not None:
            feed_dict.update({controller_initial_state_autoregressive : controller_final_state_autoregressive_cur})
        if controller_final_state_gt_cur is not None:
            feed_dict.update({controller_final_state_gt : controller_final_state_gt_cur})
        if mode == "train":
            feed_dict.update({keep_prob : KEEP_PROB_TRAIN})
            summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
                session.run([summaries,
                             optimizer,
                             mse_autoregressive_steering,
                             controller_final_state_gt,
                             controller_final_state_autoregressive],
                           feed_dict = feed_dict)
            train_writer.add_summary(summary, global_train_step)
            global_train_step += 1
        elif mode == "valid":
            model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions,
                             summaries,
                             mse_autoregressive_steering,
                             controller_final_state_autoregressive
                             ],
                           feed_dict = feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]
        elif mode == "test":
            model_predictions, controller_final_state_autoregressive_cur = \
                session.run([
                    steering_predictions,
                    controller_final_state_autoregressive
                ],
                feed_dict = feed_dict)
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]
        if mode != "test":
            acc_loss += loss
            print ('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step+1)))
    print()
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)

if __name__ == '__main__':
    best_validation_score = None
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        session.run(tf.global_variables_initializer())
        print('Initialized')
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            print("Restoring from", ckpt)
            saver.restore(sess=session, save_path=ckpt)
        for epoch in range(NUM_EPOCHS):
            print("Starting epoch %d" % epoch)
            print("Validation:")
            valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")
            if best_validation_score is None:
                best_validation_score = valid_score
            if valid_score < best_validation_score:
                saver.save(session, 'v3/checkpoint-sdc-ch2')
                best_validation_score = valid_score
                print('\r', "SAVED at epoch %d" % epoch)
                with open("v3/valid-predictions-epoch%d" % epoch, "w") as out:
                    result = np.float128(0.0)
                    for img, stats in valid_predictions.items():
                        print(img, stats, file=out)
                        result += stats[-1]
                print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)))
                with open("v3/test-predictions-epoch%d" % epoch, "w") as out:
                    _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
                    print("frame_id,steering_angle", file=out)
                    for img, pred in test_predictions.items():
                        img = img.replace("challenge_2/Test-final/center/", "")
                        print("%s,%f" % (img, pred), file=out)
            if epoch != NUM_EPOCHS - 1:
                print("Training")
                do_epoch(session=session, sequences=train_seq, mode="train")


# Basically that's it.
#
# The model can be further fine-tuned for the challenge purposes by subsetting the training set and setting the aux_cost_weight to zero. It improves the result slightly, but the improvement is marginal (doesn't affect the challenge ranking). For real-life usage it would be probably harmful because of the risk of overfitting to the dev- or even testset.
#
# Of course, speaking of realistic models, we don't need to constrain our input only to the central camera -- other cameras and sensors can dramatically improve the performance. Also it is useful to think of a realistic delay for the target sequence to make an actual non-zero-latency control possible.
#
# If something in this writeup is unclear, please write me a e-mail so that I can add the necessary comments/clarifications.
