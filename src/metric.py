from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import argparse
from skimage import data, img_as_float
from skimage import measure
import cv2
from sklearn.metrics.pairwise import cosine_similarity

## calu two image psnr
def PSNR2(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))
    #print("Mean Square Error:",np.mean(np.square(y_pred - y_true)))
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
	img.save("/home/c95lpy/image-quality-metrics/image/resizenir.png")
	x = img_to_array(img) # this is a Numpy array with shape (3, ?, ?)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, ?, ?)
	x = x.astype('float32') / 255.

	return x


"""
gt - ground truth image path
predict - predict image path

"""
class Metric ():
	def calculatePSNR(self, gt, predict):
		img1 = img2np(gt)
		img2 = img2np(predict)
		return PSNR2(img1,img2)

	def calculateSSIM(self, gt, predict):
		ssim_img1 = cv2.imread(gt, 1)
		ssim_img2 = cv2.imread(predict, 1)
		return measure.compare_ssim(ssim_img1, ssim_img2, multichannel=True)

	def calculateMSE(self, gt, predict):
		img1 = img2np(gt)
		img2 = img2np(predict)
		return np.mean(np.square(img2 - img1))

	def calculateCosSim(self, gt, predict):
		img1 = img2np(gt)
		img2 = img2np(predict)
		img1=np.reshape(img1,(1,img1.shape[1]*img1.shape[2]*img1.shape[3]))
		img2=np.reshape(img2,(1,img2.shape[1]*img2.shape[2]*img2.shape[3]))
		cos_sim = cosine_similarity(img1,img2)
		return cos_sim[0][0]


gt = "./output_org20/1.jpg"
predict = "./output_blur20/1.jpg"

M = Metric()
result = M.calculateCosSim(gt,predict)
print("cos_sim:", result)
result = M.calculateMSE(gt,predict)
print("MSE:", result)
result = M.calculatePSNR(gt,predict)
print("PSNR:", result)
result = M.calculateSSIM(gt,predict)
print("SSIM:", result)
