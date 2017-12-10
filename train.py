import tensorflow as tf
import numpy as np
import os
from config import *
from data_utils import *
from models.models import *

# # Model
graph = tf.Graph()
with graph.as_default():
    # Build model
    # model = Komada(graph, mean, std)
    model_type = CNN
    model_dir = "cnn"

    if model_type is CNN:
        (train_seq_X, train_seq_Y, valid_seq_X, valid_seq_Y), (mean, std) = process_csv_cnn(filename="./data/train/output/interpolated.csv", val=5) # concatenated interpolated.csv from rosbags
        test_seq_X, test_seq_Y = read_csv("./data/test/final_example.csv", train=False, cnn=True) # interpolated.csv for testset filled with dummy values

    else:
        (train_seq_X, valid_seq_X), (mean, std) = process_csv(filename="./data/train/output/interpolated.csv", val=5) # concatenated interpolated.csv from rosbags
        test_seq_X = read_csv("./data/test/final_example.csv", train=False) # interpolated.csv for testset filled with dummy values

        train_seq_Y, valid_seq_Y, test_seq_Y = None, None, None

    model = model_type(graph, mean, std)

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

    checkpoint_dir = os.getcwd() + "/%s" % model_dir

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
            if epoch != NUM_EPOCHS - 1:
                print("Training")
                model.do_epoch(session=session, sequences=train_seq_X, labels=train_seq_Y, mode="train")
            print("Validation:")
            valid_score, valid_predictions = model.do_epoch(session=session, sequences=valid_seq_X, labels=valid_seq_Y, mode="valid")
            if best_validation_score is None:
                best_validation_score = valid_score
            if valid_score < best_validation_score:
                model.saver.save(session, '{}/checkpoint-sdc-ch2'.format(model_dir))
                best_validation_score = valid_score
                print('\r', "SAVED at epoch %d" % epoch)
                with open("%s/valid-predictions-epoch%d" % model_dir, epoch, "w") as out:
                    result = np.float128(0.0)
                    for img, stats in valid_predictions.items():
                        print(img, stats, file=out)
                        result += stats[-1]
                print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)))
                with open("%s/test-predictions-epoch%d" % model_dir, epoch, "w") as out:
                    _, test_predictions = model.do_epoch(session=session, sequences=test_seq_X, labels=test_seq_Y, mode="test")
                    print("frame_id,steering_angle", file=out)
                    for img, pred in test_predictions.items():
                        img = img.replace("challenge_2/Test-final/center/", "")
                        print("%s,%f" % (img, pred), file=out)
