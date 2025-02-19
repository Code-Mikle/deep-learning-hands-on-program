"""
    加载数据集
"""

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from config.config import conf

# conf_data = OmegaConf.load('config/config.yaml')
# OmegaConf.to_yaml(conf_data, resolve=True)


training_data = datasets.CIFAR10(root=conf.dataset_path, train=True, download=True, transform=ToTensor())
testing_data = datasets.CIFAR10(root=conf.dataset_path, train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=conf.hyper_parameter.batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=conf.hyper_parameter.batch_size, shuffle=True)

if __name__ == '__main__':
    # labels_map = {
    #     0: 'airplane',
    #     1: 'automobile',
    #     2: 'bird',
    #     3: 'cat',
    #     4: 'deer',
    #     5: 'dog',
    #     6: 'frog',
    #     7: 'horse',
    #     8: 'ship',
    #     9: 'truck'
    # }
    #
    # figure = plt.figure(figsize=(10, 10))
    # cols, rows = 5, 5
    # for i in range(cols * rows):
    #     sample_idx = torch.randint(0, len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i + 1)
    #     plt.title(labels_map[label])
    #     plt.axis('off')
    #     plt.imshow(img.permute(1, 2, 0))
    # plt.show()
    print(train_dataloader)
    print("asd")
