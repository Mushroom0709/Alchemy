import os

import matplotlib.pyplot as plt
import mindspore.dataset as ds
import numpy as np
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """
       create dataset for train or test

       Args:
           data_path (str): Data path
           batch_size (int): The number of data records in each group
           repeat_size (int): The number of replicated data records
           num_parallel_workers (int): The number of parallel workers
    """

    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define some parameters needed for data enhancement and rough justification
    resize_height = 32
    resize_width = 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1.0 * 0.1307 / 0.3081

    # according to the parameters, generate the corresponding data enhancement method
    resize_op = CV.Resize((resize_height, resize_width),interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # using map to apply operations to dataset
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # process the generated dataset
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


if __name__ == "__main__":
    train_data_path = "./MNIST/train"
    test_data_path = "./MNIST/test"

    ms_dataset = create_dataset(train_data_path)
    print('Number of groups in the dataset:', ms_dataset.get_dataset_size())

    data = next(ms_dataset.create_dict_iterator(output_numpy=True))
    images = data["image"]
    labels = data["label"]
    print('Tensor of image', images.shape)
    print('Labels:', labels)

    count = 1
    for i in images:
        plt.subplot(4, 8, count)
        plt.imshow(np.squeeze(i))
        plt.title('num:%s' % labels[count-1])
        plt.xticks([])
        count += 1
        plt.axis("off")
    plt.show()
