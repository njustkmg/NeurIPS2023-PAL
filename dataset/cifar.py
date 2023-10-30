import logging
import math
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

__all__ = ['cifar10_mean', 'cifar10_std', 'cifar100_mean', 'cifar100_std', 'normal_mean', 'normal_std','Use_Subset']
### Enter Path of the data directory.
DATA_PATH = './data'


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


class Use_Subset(Subset):
    def __init__(self, dataset, indices, change=False, classes = 6):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices
        self.data = dataset.data
        self.targets = dataset.targets
        self.change = change
        self.classes = classes
        
    def __getitem__(self, idx):
        feature, targets, index = self.dataset[self.indices[idx]]
        if self.change:
            if int(targets) < self.classes:
                targets = self.classes
            else:
                targets = self.classes + 1
        return feature, targets, index

    def __len__(self):
        return len(self.indices)



def get_cifar(args, train_labeled_idxs=None, train_unlabeled_idxs=None, val_idxs = None, norm=True):
    root = args.root
    name = args.dataset
    if name == "cifar10":
        data_folder = datasets.CIFAR10
        data_folder_main = CIFAR10SSL
        mean = cifar10_mean
        std = cifar10_std
        num_class = 10
    elif name == "cifar100":
        data_folder = datasets.CIFAR100
        data_folder_main = CIFAR100SSL
        mean = cifar100_mean
        std = cifar100_std
        num_class = 100

    else:
        raise NotImplementedError()
    assert num_class > args.num_classes

    base_dataset = data_folder(root, train=True, download=True)
    base_dataset.targets = np.array(base_dataset.targets)

     # get the train idx 
    print('-'*40 + ' Get Ids ' + '-'*40)
    if train_labeled_idxs == None and train_unlabeled_idxs == None and val_idxs == None:
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
            x_u_split(args, base_dataset.targets)
    else:
        train_labeled_idxs = list(train_labeled_idxs)
        train_unlabeled_idxs = list(train_unlabeled_idxs)
        val_idxs = list(val_idxs)
    print('-'*40 + ' Finished ' + '-'*40)
    tar_len = list(range(len(list(base_dataset.targets))))

    norm_func = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # dateset ready
    base_datasets = data_folder_main(root, train=True, 
            indexs=tar_len,
            transform=norm_func)

    train_labeled_dataset = data_folder_main(
        root, train_labeled_idxs, train=True,
        transform=norm_func)
    
    train_unlabeled_dataset = data_folder_main(
        root, train_unlabeled_idxs, train=True,
        transform=norm_func, return_idx=False)
    
    val_dataset = data_folder_main(
        root, val_idxs, train=True,
        transform=norm_func)
    
    test_dataset = data_folder(
        root, train=False, transform=norm_func_test, download=False)
    
    test_dataset.targets = np.array(test_dataset.targets)
    target_ind = np.where(test_dataset.targets >= args.num_classes)[0]
    test_dataset.targets[target_ind] = args.num_classes

    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    
    print("Dataset: %s"%name)
    print(f"Labeled examples: {len(unique_labeled)} "
                
                f"Unlabeled examples: {len(train_unlabeled_idxs)} "

                f"Valdation samples: {len(val_labeled)}")
    
    return list(unique_labeled), list(train_unlabeled_idxs), val_idxs, base_datasets, train_labeled_dataset, train_unlabeled_dataset, \
           test_dataset, val_dataset

 
def x_u_split(args, labels):
    label_per_class = args.num_labeled #// args.num_classes
    val_per_class = args.num_val #// args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class+val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled * args.num_classes
    
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * 25 / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    unlabeled_idx = np.array(range(len(labels)))
    unlabeled_idx = list(set(unlabeled_idx) - set(labeled_idx))
    unlabeled_idx = list(set(unlabeled_idx) - set(val_idx))
    return labeled_idx, unlabeled_idx, val_idx


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


def get_transform(mean, std, image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.ToTensor()
    return train_transform, test_transform



DATASET_GETTERS = {'cifar10': get_cifar,
                   'cifar100': get_cifar,
                   }
