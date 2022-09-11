import numpy as np
import torchkeras
from network_train import get_device
from network import RepSharingKernelNet, SmapeLoss
import torch
from torchkeras.metrics import Accuracy
from data_handler import DataHandler
import torch.utils.data as Data
from image_utils import save_image, batch_psnr, batch_ssim
import time


def test():
    learn_rate = 0.001
    device = get_device()

    rep_sharing_net = RepSharingKernelNet(device)
    model_clone = torchkeras.KerasModel(net=rep_sharing_net,
                                        loss_fn=SmapeLoss(),
                                        optimizer=torch.optim.Adam(rep_sharing_net.parameters(), lr=learn_rate),
                                        metrics_dict={'acc': Accuracy()})
    model_clone.to(device)
    model_clone.net.load_state_dict(torch.load('models/checkpoint.pt'))

    test_data = DataHandler(data_dir='datasets',
                            subset='test',
                            scene_list=['classroom'],
                            patch_per_img=0,
                            patch_width=128,
                            patch_height=128)
    test_data = Data.DataLoader(test_data.dataset,
                                batch_size=60,
                                shuffle=False,
                                num_workers=1)
    # 计算ssim， psnr
    predicts = model_clone.predict(test_data).numpy()
    reals = np.zeros(predicts.shape)
    noisy_list = []

    i = 0
    for noisy, real in test_data:
        save_image(predicts[i], 'test_results/test' + str(i) + '.png')
        reals[i] = real.numpy()
        noisy_list.append(noisy)
        i += 1

    ssim = batch_ssim(reals, predicts)
    psnr = batch_psnr(reals, predicts)

    print('ssim:', ssim)
    print('psnr:', psnr)

    # 计算每帧去噪平均时间
    total_time = 0

    for i in range(len(noisy_list)):
        noisy = noisy_list[i]
        start = time.perf_counter()
        model_clone.forward(noisy)
        end = time.perf_counter()
        total_time += end - start

    mean_time = total_time / len(noisy_list)

    print('time per frame:', mean_time)
