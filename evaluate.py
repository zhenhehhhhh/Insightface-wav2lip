import skimage.metrics as skm
import numpy as np
import os
import cv2

def mse(ori, pro):
    mse = np.mean( (ori - pro) ** 2 )
    return mse

def ssim(ori, pro):
    ssim_val = skm.structural_similarity(ori, pro, multichannel=True)
    return float(ssim_val)

def psnr(ori, pro):
    mse = np.mean((pro - ori)**2)
    if mse == 0:
        return float('inf')
    else:
        psnr = 20 * np.log10(255 / np.sqrt(mse))
        return float(psnr)
 
if __name__ == '__main__':
    image_path = 'photos/'
    imgs_orig = os.listdir(image_path+'initial')
    f = open("evalute.log", "w+", encoding='utf-8')
    MSE_ALL = 0
    SSIM_ALL = 0
    PSNR_ALL = 0

    MSE_MAX, MSE_MIN = 0, 100
    SSIM_MAX, SSIM_MIN = 0, 1
    PSNR_MAX, PSNR_MIN = 0, 100
    for i in range(len(imgs_orig)):
        img_pro_path = image_path + 'process/' + 'frame_d{}.jpg'.format(i)
        img_ori_path = image_path + 'initial/' + 'frame_d_raw{}.jpg'.format(i)
        img = cv2.imread(img_pro_path)
        img_ori = cv2.imread(img_ori_path)

        MSE = mse(img_ori, img)
        SSIM = ssim(img_ori, img)
        PSNR = psnr(img_ori, img)

        MSE_ALL += MSE
        SSIM_ALL += SSIM
        PSNR_ALL += PSNR

        if MSE > MSE_MAX:
            MSE_MAX = MSE
        if MSE < MSE_MIN:
            MSE_MIN = MSE

        if SSIM > SSIM_MAX:
            SSIM_MAX = SSIM
        if SSIM < SSIM_MIN:
            SSIM_MIN = SSIM

        if PSNR > PSNR_MAX:
            PSNR_MAX = PSNR
        if PSNR < PSNR_MIN:
            PSNR_MIN = PSNR
        
        f.write(str(i) + "    " + str(MSE) + "   " + str(SSIM) + "   " + str(PSNR) + '\n')
    MSE_AVER = MSE_ALL / len(imgs_orig)
    SSIM_AVER = SSIM_ALL / len(imgs_orig)
    PSNR_AVER = PSNR_ALL / len(imgs_orig)
    f.write("MAX    " + str(MSE_MAX) + "   " + str(SSIM_MAX) + "   " + str(PSNR_MAX) + '\n')
    f.write("MIN    " + str(MSE_MIN) + "   " + str(SSIM_MIN) + "   " + str(PSNR_MIN) + '\n')
    f.write("AVER   " + str(MSE_AVER) + "   " + str(SSIM_AVER) + "   " + str(PSNR_AVER) + '\n')

    f.close()