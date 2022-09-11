import OpenEXR
import numpy as np
import Imath
import torch
import PIL.Image
from skimage import metrics


def load_exr(filename, datatype=np.float16, axis=-1):
    if not OpenEXR.isOpenExrFile(filename):
        raise Exception("File '%s' is not an EXR file." % filename)
    infile = OpenEXR.InputFile(filename)

    header = infile.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    matrix_ch_B = np.fromstring(infile.channels('B')[0], dtype=datatype).reshape(height, width)
    matrix_ch_G = np.fromstring(infile.channels('G')[0], dtype=datatype).reshape(height, width)
    matrix_ch_R = np.fromstring(infile.channels('R')[0], dtype=datatype).reshape(height, width)

    matrix_new = np.stack((matrix_ch_R, matrix_ch_G, matrix_ch_B), axis=axis)
    return matrix_new


def save_exr(image, filename, datatype=np.float16):
    HALF = Imath.PixelType(Imath.PixelType.HALF)

    data = image.astype(datatype)
    channels = {}
    channel_data = {}
    channel_name = 'B'
    channels['B'] = Imath.Channel(type=HALF)
    channel_data[channel_name] = data[:, :, 2].tostring()
    channel_name = 'G'
    channels['G'] = Imath.Channel(type=HALF)
    channel_data[channel_name] = data[:, :, 1].tostring()
    channel_name = 'R'
    channels['R'] = Imath.Channel(type=HALF)
    channel_data[channel_name] = data[:, :, 0].tostring()

    new_header = OpenEXR.Header(data.shape[1], data.shape[0])
    new_header['channels'] = channels
    out = OpenEXR.OutputFile(filename, new_header)
    out.writePixels(channel_data)


def load_exr_one_channel(filename, datatype=np.float16):
    if not OpenEXR.isOpenExrFile(filename):
        raise Exception("One Channel: File '%s' is not an EXR file." % filename)

    infile = OpenEXR.InputFile(filename)
    header = infile.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    matrix_ch_G = np.fromstring(infile.channels('G')[0], dtype=datatype).reshape(height, width)
    return matrix_ch_G


def save_exr_one_channel(image, filename, datatype=np.float16):
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    data = image.astype(datatype)

    channel_name = 'G'
    channels = {channel_name: Imath.Channel(type=HALF)}
    channel_data = {channel_name: data.tostring()}

    new_header = OpenEXR.Header(data.shape[1], data.shape[0])
    new_header['channels'] = channels
    out = OpenEXR.OutputFile(filename, new_header)
    out.writePixels(channel_data)


def bmfr_gamma_correction(color) -> torch.Tensor:
    gamma = 0.45
    correction_color = color / torch.max(color)
    correction_color = torch.pow(correction_color, gamma)
    # print(correction_color)
    # print(torch.max(correction_color), torch.min(correction_color))
    # correction_color = torch.clamp(correction_color, 0, 1)
    return correction_color


def save_image(image, filename, mode=None):
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = clip_to_uint8(image)
    else:
        assert image.dtype == np.uint8
        image.astype(np.uint8)
    PIL.Image.fromarray(image, mode=mode).save(filename)


def clip_to_uint8(arr):
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def batch_psnr(img_true, img_test):
    psnr_arr = np.zeros(img_true.shape[0])
    for idx in range(img_true.shape[0]):
        psnr_arr[idx] = metrics.peak_signal_noise_ratio(
            img_true[idx, :, :, :], img_test[idx, :, :, :])
    return psnr_arr, np.mean(psnr_arr)


def batch_ssim(img_true, img_test, mc=True):
    ssim_arr = np.zeros(img_true.shape[0])
    for idx in range(img_true.shape[0]):
        ssim_arr[idx] = metrics.structural_similarity(
            img_true[idx, :, :, :], img_test[idx, :, :, :], multichannel=mc)
    return ssim_arr, np.mean(ssim_arr)
