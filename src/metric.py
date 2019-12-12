from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import argparse
from skimage import data, img_as_float
from skimage import measure
import cv2

## calu two image psnr
def PSNR2(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))
    print("Mean Square Error:",np.mean(np.square(y_pred - y_true)))
    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

## image file read to np array
def img2np(filename):
	img = load_img(filename)# this is a PIL image
	x = img_to_array(img) # this is a Numpy array with shape (3, ?, ?)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, ?, ?)
	x = x.astype('float32') / 255.

	return x

def imgresize2np(filename):
	img = load_img(filename,target_size=(args.predict_h, args.predict_w))# this is a PIL image
	img.save("../image/resizenir.png")
	x = img_to_array(img) # this is a Numpy array with shape (3, ?, ?)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, ?, ?)
	x = x.astype('float32') / 255.

	return x

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--gt', type=str, default="../image/org.png")
parser.add_argument('--target', type=str, default="../image/blur.png")
parser.add_argument('--predict_w', type=int, default=300)
parser.add_argument('--predict_h', type=int, default=168)

args = parser.parse_args()

gt = args.gt
predict = args.target

#compare of psnr
img1 = imgresize2np(gt)
img2 = img2np(predict)
print("PSNR:", PSNR2(img1,img2),"(dB)")

#compare of ssim
img_path = "../image/resizenir.png"
ssim_img1 = cv2.imread(img_path, 0)
img_path = args.target
ssim_img2 = cv2.imread(img_path, 0)
ssim_none = measure.compare_ssim(ssim_img1, ssim_img2)
print("SSIM:", ssim_none)
