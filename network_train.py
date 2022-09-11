import random
import numpy as np
import torch
from network import RepSharingKernelNet, SmapeLoss
from data_handler import DataHandler
import torch.utils.data as Data
import torchkeras
from torchkeras.metrics import Accuracy
import pickle


def init_seed(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    # if seed == 0:
    #     # if True, causes cuDNN to only use deterministic convolution algorithms.
    #     torch.backends.cudnn.deterministic = True
    #     # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    #     torch.backends.cudnn.benchmark = False


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device


def train():
    batch_size = 64
    learn_rate = 0.001
    epoch = 300
    device = get_device()

    init_seed(2022)

    data_dir = 'data'
    scenes = ['livingroom', 'sanmiguel', 'sponza', 'sponzaglossy', 'sponzamovinglight']

    print('加载训练集')
    train_data = DataHandler(data_dir=data_dir,
                             subset='train',
                             scene_list=scenes,
                             patch_per_img=80,
                             patch_width=128,
                             patch_height=128)
    train_data = Data.DataLoader(train_data.dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2)

    print('加载验证集')
    valid_data = DataHandler(data_dir=data_dir,
                             subset='valid',
                             scene_list=scenes,
                             patch_per_img=20,
                             patch_width=128,
                             patch_height=128)
    valid_data = Data.DataLoader(valid_data.dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    rep_sharing_net = RepSharingKernelNet(device)
    model = torchkeras.KerasModel(net=rep_sharing_net,
                                  loss_fn=SmapeLoss(),
                                  optimizer=torch.optim.Adam(rep_sharing_net.parameters(), lr=learn_rate),
                                  metrics_dict={'acc': Accuracy()})
    model.to(device)

    df_history = model.fit(epochs=epoch,
                           train_data=train_data,
                           val_data=valid_data,
                           patience=20,
                           monitor="val_acc",
                           mode="max",
                           ckpt_path='models/checkpoint.pt')

    with open('history.pkl', 'wb') as f:
        pickle.dump(df_history, f)


if __name__ == '__main__':
    train()
