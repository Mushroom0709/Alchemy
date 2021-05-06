import os
from create_dataset import create_dataset
from lenet5 import LeNet5
from step_loss_acc_info import StepLossAccInfo

import matplotlib.pyplot as plt
import mindspore.nn as nn
from mindspore.nn import Accuracy, SoftmaxCrossEntropyWithLogits
from mindspore import Tensor, Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor


def eval_show(p_steps_eval):
    plt.xlabel("step number")
    plt.ylabel("Model accuracy")
    plt.title("Model accuracy variation chart")
    plt.plot(p_steps_eval["step"], p_steps_eval["acc"], "red")
    plt.show()


def loss_show(p_steps_loss):
    steps = p_steps_loss["step"]
    loss_value = p_steps_loss["loss_value"]
    steps = list(map(int, steps))
    loss_value = list(map(float, loss_value))
    plt.plot(steps, loss_value, color="red")
    plt.xlabel("Steps")
    plt.ylabel("Loss_value")
    plt.title("Change chart of model loss value")
    plt.show()


if __name__ == "__main__":
    lr = 0.01
    momentum = 0.9

    # create the network
    network = LeNet5()

    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    epoch_size = 5
    mnist_path = "./MNIST/"
    model_path = "./model/ckpt/mindspore_quick_start/"

    repeat_size = 1
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32, repeat_size)
    ds_eval = create_dataset(os.path.join(mnist_path, "test"), 32)

    # clean up old run files before in Linux
    os.system('rm -rf {0}*.ckpt {0}*.meta {0}*.pb'.format(model_path))

    # define the model
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # save the network model and parameters for subsquenece fine-tuning
    config_ck = CheckpointConfig(save_checkpoint_steps=375, keep_checkpoint_max=16)

    # group layers into an object whith tarining and evaluation features
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=model_path, config=config_ck)

    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}

    # collect the steps,loss and accuracy infofmation
    step_loss_acc_info = StepLossAccInfo(model, ds_eval, steps_loss, steps_eval)

    model.train(epoch_size,
                ds_train,
                callbacks=[ckpoint_cb, LossMonitor(125), step_loss_acc_info],
                dataset_sink_mode=False)

    loss_show(steps_loss)
    eval_show(steps_eval)


