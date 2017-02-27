import pandas as pd

import theano
from theano import tensor as T
from sklearn.decomposition import PCA

import numpy as np

"""
utility functions for the train_network main function
"""

def print_classes_accuracy(cls_accs):
    for idx, cls_accuracy in enumerate(cls_accs):
        print ("For class index " + str(idx) + ": ", end="")
        print ("Accuracy: ", cls_accuracy)

def make_output_arrays(output_nums):
    output_arrs = np.zeros((len(output_nums), len(set(output_nums))))
    for idx, num in enumerate(output_nums):
        output_arrs[idx][num] = 1.0

    return output_arrs

def get_data(filepath, cross_val_k, curr_iter):

    df = pd.read_csv(filepath, header=None)
    df_inputs = df[list(range(len(df.columns) - 1))]
    df_outputs = df[len(df.columns) - 1]

    start_idx = int(curr_iter * (len(df)/cross_val_k))
    end_idx = start_idx + int(len(df)/cross_val_k)

    test_indexes = list(range(start_idx, end_idx))
    train_indexes = list(range(0, start_idx)) + list(range(end_idx, len(df)))

    df_outputs.replace("Slight-Right-Turn", 0, inplace=True)
    df_outputs.replace("Sharp-Right-Turn", 1, inplace=True)
    df_outputs.replace("Slight-Left-Turn", 2, inplace=True)
    df_outputs.replace("Move-Forward", 3, inplace=True)

    train_in = df_inputs.ix[train_indexes].values.astype(np.float32)
    train_out = df_outputs.ix[train_indexes].values.astype(np.int32)

    test_in = df_inputs.ix[test_indexes].values.astype(np.float32)
    test_out = df_outputs.ix[test_indexes].values.astype(np.int32)

    return (train_in, train_out), (test_in, test_out)


def train_network(k, printout_period, source_filepath, iters):
    """
    trains a simple MLP using k-fold cross-validation

    params:
    k - the 'k' in k-fold
    printout_period - how many iterations must pass for a printout of cross entropy,
                        prediction accuracy, etc. since the last printout
    source_filepath - filepath of data
    iters - amount of iterations (epochs) to train the MLP on per cross-validation iteration
    """

    test_accuracies = []
    for i in range(k):
        train_data, test_data = get_data(source_filepath, k, i)

        train_inputs = train_data[0]
        train_outputs_acc = train_data[1]

        test_inputs = test_data[0]
        test_outputs_acc = test_data[1]

        train_outputs = make_output_arrays(train_outputs_acc)
        test_outputs = make_output_arrays(test_outputs_acc)

        #pca_obj = PCA(n_components=16)
        #train_inputs = pca_obj.fit_transform(train_inputs)
        #test_inputs = pca_obj.transform(test_inputs)

        num_hidden = 4

        x = T.dmatrix('x')
        y = T.dmatrix('y')
        y_acc = T.dmatrix('y_acc')
        num_inst = T.scalar('num_inst')

        test_x = T.dmatrix('test_x')

        test_x_output = T.argmax(test_x, axis=1)
        test_x_sum = T.sum(test_x, axis=1)

        alpha = 2e-2
        mu = 0.9

        w1 = theano.shared(np.random.randn(len(train_inputs[0]), num_hidden))
        w2 = theano.shared(np.random.randn(num_hidden, len(train_outputs[0])))

        b1 = theano.shared(np.random.randn(num_hidden))
        b2 = theano.shared(np.random.randn(len(train_outputs[0])))

        w1_update = theano.shared(np.zeros((len(train_inputs[0]), num_hidden)))
        w2_update = theano.shared(np.zeros((num_hidden, len(train_outputs[0]))))
        b1_update = theano.shared(np.zeros((num_hidden)))
        b2_update = theano.shared(np.zeros((len(train_outputs[0]))))

        hidden_inputs = T.dot(x, w1) + b1
        hidden_outputs = 1/(1 + T.exp(-hidden_inputs))
        output = T.dot(hidden_outputs, w2) + b2

        class_preds = T.argmax(output, axis=1)
        class_preds = class_preds.reshape((class_preds[0], 1))
        perc_correct = T.sum(T.eq(class_preds, y_acc))/num_inst

        all_exp = T.exp(output)
        softmax_sums = T.sum(all_exp, axis=1).reshape((all_exp.shape[0], 1))
        softmax = all_exp / softmax_sums

        x_entropy = -T.sum(T.log(softmax) * y)/y.shape[0]

        gw1, gw2, gb1, gb2 = T.grad(x_entropy, [w1, w2, b1, b2])

        def class_accuracy(cls_id):
            cls_idxes = T.eq(y_acc, cls_id).nonzero()[0]
            num_eq = T.sum(T.eq(y_acc[cls_idxes], class_preds[cls_idxes]))
            return num_eq/cls_idxes.shape[0]

        classes_accuracy, junk = theano.scan(fn=class_accuracy, sequences=[T.arange(4)])

        train = theano.function(inputs=[x, y, y_acc, num_inst], outputs=[x_entropy, perc_correct, classes_accuracy], updates=[(w1_update, mu * w1_update - gw1 * alpha), 
            (w2_update, mu * w2_update - gw2 * alpha), (b1_update, mu * b1_update - gb1 * alpha), (b2_update, mu * b2_update - gb2 * alpha),
            (w1, w1 + w1_update), (w2, w2 + w2_update), (b1, b1 + b1_update), (b2, b2 + b2_update)])

        test_network = theano.function(inputs=[x, y_acc, num_inst], outputs=[perc_correct, classes_accuracy])

        entropy_list = []
        for iteration in range(iters):
            info = train(train_inputs, train_outputs, train_outputs_acc.reshape(len(train_outputs_acc), 1), len(train_outputs_acc))
            entropy_list.append(info[0])
            if iteration % printout_period == 0:
                print ("Iteration: ", iteration)
                print ("Cross entropy: ", info[0])
                print ("Total accuracy: ", info[1])
                print ("Accuracy for each class:")
                print_classes_accuracy(info[2])
                print ("-----------------------------")
            if (len(entropy_list) > 2 and entropy_list[-1] > entropy_list[-2]):
                alpha *= 0.999
            else:
                alpha *= 1.0001

        info = test_network(test_inputs, test_outputs_acc.reshape(len(test_outputs_acc), 1), len(test_outputs_acc))

        print ("Overall test accuracy: " + str(info[0]))
        print ("Accuracy per class: ")
        print_classes_accuracy(info[1])

        test_accuracies.append(info[0])

        print ("=========================")

    print ("K-fold cross-validation accuracy results: ")
    for iteration, acc in enumerate(test_accuracies):
        print ("Iteration " + str(iteration + 1) + ": " + str(acc))

if __name__ == "__main__":
    train_network(5, 300, "sensor_readings_4.data", 4000)
