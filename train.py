import tensorflow as tf
import numpy as np
import os
from config import *
from data_utils import process_csv, read_csv

from models.komada import Komada

(train_seq, valid_seq), (mean, std) = process_csv(filename="./data/train/interpolated.csv", val=5) # concatenated interpolated.csv from rosbags
test_seq = read_csv("./data/test/final_example.csv", train=False) # interpolated.csv for testset filled with dummy values

# # Model
graph = tf.Graph()
with graph.as_default():
    # Build model
    model = Komada(graph, mean, std)


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

    best_validation_score = None
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        print('Initialized')
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            print("Restoring from", ckpt)
            model.saver.restore(sess=session, save_path=ckpt)
        for epoch in range(NUM_EPOCHS):
            print("Starting epoch %d" % epoch)
            print("Validation:")
            valid_score, valid_predictions = model.do_epoch(session=session, sequences=valid_seq, mode="valid")
            if best_validation_score is None:
                best_validation_score = valid_score
            if valid_score < best_validation_score:
                model.saver.save(session, 'v3/checkpoint-sdc-ch2')
                best_validation_score = valid_score
                print('\r', "SAVED at epoch %d" % epoch)
                with open("v3/valid-predictions-epoch%d" % epoch, "w") as out:
                    result = np.float128(0.0)
                    for img, stats in valid_predictions.items():
                        print(img, stats, file=out)
                        result += stats[-1]
                print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)))
                with open("v3/test-predictions-epoch%d" % epoch, "w") as out:
                    _, test_predictions = model.do_epoch(session=session, sequences=test_seq, mode="test")
                    print("frame_id,steering_angle", file=out)
                    for img, pred in test_predictions.items():
                        img = img.replace("challenge_2/Test-final/center/", "")
                        print("%s,%f" % (img, pred), file=out)
            if epoch != NUM_EPOCHS - 1:
                print("Training")
                model.do_epoch(session=session, sequences=train_seq, mode="train")
