import os.path
import matplotlib.pyplot as plt
import numpy

from lenet5 import LeNet5

import mindspore.nn as nn
from mindspore.nn import Accuracy, SoftmaxCrossEntropyWithLogits
from mindspore import Tensor, Model, load_checkpoint, load_param_into_net
from create_dataset import create_dataset


# testing relate modules
def test_net(network, model, mnist_path):
    """Define the evaluation method"""

    print("==================== Starting Testing ===============")
    param_dict = load_checkpoint("./model/ckpt/mindspore_quick_start/checkpoint_lenet-1_1874.ckpt")
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(mnist_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("==================== Accuracy:{} ===============".format(acc))


def test_inference(model, test_data_path):
    ds_test = create_dataset(test_data_path).create_dict_iterator()
    data = next(ds_test)
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    output = model.predict(Tensor(data["image"]))
    pred = numpy.argmax(output.asnumpy(), axis=1)
    err_num = []
    index = 1
    for i in range(len(labels)):
        plt.subplot(4, 8, i+1)
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title("pre:{}".format(pred[i]), color=color)
        plt.imshow(numpy.squeeze(images[i]))
        plt.axis("off")
        if color == 'red':
            index = 0
            print(
                "Row {}, column {} is incorrectly identified as {}, the correct value should be {}".format(
                    int(i/8)+1,
                    i % 8+1,
                    pred[i],
                    labels[i]),
                '\n'
            )
    if index:
        print("All the figures in this group are predicted correctly")
    print(pred, "<--Predicted figures")
    print(labels, "<--The right number")
    plt.show()


if __name__ == "__main__":
    mnist_path = "./MNIST/"
    test_data_path = "./MNIST/test/"
    model_path = "./model/ckpt/mindspore_quick_start/"

    lr = 0.01
    momentum = 0.9

    # create the network
    network = LeNet5()
    param_dict = load_checkpoint("./model/ckpt/mindspore_quick_start/checkpoint_lenet-5_1500.ckpt")
    load_param_into_net(network, param_dict)

    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # test_net(network, model, mnist_path)
    test_inference(model, test_data_path)
