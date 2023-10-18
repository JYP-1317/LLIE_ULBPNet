from __future__ import print_function
import argparse
import torch
from model import DLN
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir
import os
import lpips
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=128, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')  # batch size 영향
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='test_img')
parser.add_argument('--model_type', type=str, default='DLN')
parser.add_argument('--output', default='./output/', help='Location to save checkpoint models')
parser.add_argument('--modelfile', default='models/DLN_5743.pth', help='sr pretrained base model')
parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
parser.add_argument('--chop', type=bool, default=False)

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)

model = DLN()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(
    opt.modelfile,
    map_location=lambda storage, loc: storage))
if cuda:
    model = model.cuda(gpus_list[0])
    mdoel = model.type('torch.cuda.HalfTensor')


def eval():
    model.eval()
    #TestFolder = ['DICM', 'LIME', 'LOL', 'Fusion', 'MEF', 'VV']#['LOL']
    TestFolder = ['LOL']
    # TestFolder = ['ablation']
    for i in range(TestFolder.__len__()):
        LL_filename = os.path.join(opt.image_dataset, TestFolder[i])  # LOL DCIM LIME VV Fusion NASA
        est_filename = os.path.join(opt.output, TestFolder[i])
        try:
            os.stat(est_filename)
        except:
            os.mkdir(est_filename)
        LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
        print(LL_filename)
        Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
        print(Est_img)
        trans = transforms.ToTensor()
        channel_swap = (1, 2, 0)
        time_ave = 0
        for i in range(LL_image.__len__()):
            with torch.no_grad():
                LL_in = Image.open(LL_image[i]).convert('RGB')

                # LL_in = LL_in1.resize((384,384))

                LL_int = trans(LL_in)
                c, h, w = LL_int.shape
                h_tmp = (h % 8)
                w_tmp = (w % 8)

                LL_in = LL_in.crop((0, 0, w - w_tmp, h - h_tmp))

                LL = trans(LL_in)
                LL = trans(LL_in)
                LL = LL.unsqueeze(0)
                LL = LL.cuda()
                LL = LL.type('torch.cuda.HalfTensor')
                t0 = time.time()
                prediction = model(LL)
                t1 = time.time()
                time_ave += (t1 - t0)
                prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)

                prediction = prediction * 255.0
                prediction = prediction.clip(0, 255)
                Image.fromarray(np.uint8(prediction)).save(Est_img[i])
                # prediction = prediction.resize((LL_in1.size[0],LL_in1.size[1]))

                # a= Image.fromarray(np.uint8(prediction))#.save(Est_img[i])
                # a.resize((LL_in1.size[0],LL_in1.size[1])).save(Est_img[i])
                print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))

        test_pred_folder = "datasets/LOL/test/lpips_test"  #prediction
        test_NL_folder = "datasets/LOL/test/high/" #Normal

        test_pred_list = [join(test_pred_folder, x) for x in sorted(listdir(test_pred_folder)) if is_image_file(x)]
        test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_pred_folder)) if is_image_file(x)]

        lpips_score = 0

        for i in range(test_NL_list.__len__()):
            gt = cv2.imread(test_NL_list[i])
            #gt = cv2.resize(gt, dsize=(599,398)) #LL_NET LPIPS resize
            #gt = cv2.resize(gt, dsize=(516, 388))  # DSLR LPIPS resize
            pred = cv2.imread(test_pred_list[i])

            gt_new = np.transpose(gt, [2, 0, 1])
            gt_new = np.expand_dims(gt_new, axis=0)
            gt_new = torch.tensor(gt_new) / 127.5 - 1

            pred_new = np.transpose(pred, [2, 0, 1])
            pred_new = np.expand_dims(pred_new, axis=0)
            pred_new = torch.tensor(pred_new) / 127.5 - 1

            loss_fn = lpips.LPIPS(net='alex')
            lpips_val = loss_fn.forward(pred_new, gt_new)
            lpips_score = lpips_score + lpips_val
        lpips_score = lpips_score / (test_NL_list.__len__())

        print("-------------------------------------------------")
        print("s : %d", lpips_score)

        print("===> Processing Time: %.4f ms." % (time_ave / LL_image.__len__() * 1000))


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))

    return img


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        float32, [0, 255]
        float32, [0, 255]
    '''
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()

    return rlt


def PSNR(pred, gt, shave_border):
    pred = pred[shave_border:-shave_border, shave_border:-shave_border]
    gt = gt[shave_border:-shave_border, shave_border:-shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

##Eval Start!!!!
eval()