from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import  array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.metrics.pairwise import cosine_similarity
from skimage import data
from skimage import img_as_float
from skimage import measure
import numpy as np
import argparse
import cv2

def PSNR2(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                             " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))
    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

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


class Metric ():
	"""There are four ways to measure metrics

	Set the gt / predict image path:
	>>> gt = "/home/c95lpy/image-quality-metrics/output_org20/1.jpg"
	>>> predict = "/home/c95lpy/image-quality-metrics/output_blur20/1.jpg"

	To use:
	>>> M = Metric()
	>>> result = M.calculateCosSim(gt,predict)
	>>> print("cos_sim:", result)
	0.978
	>>> result = M.calculateMSE(gt,predict)
	>>> print("MSE:", result)
	0.0078
	>>> result = M.calculatePSNR(gt,predict)
	>>> print("PSNR:", result)
	28.97
	>>> result = M.calculateSSIM(gt,predict)
	>>> print("SSIM:", result)
	0.87
	"""
	def calculatePSNR(self, gt, predict):
		"""Calculate PSNR function between two images.

		Args:
			gt: ground truth image path.
			predict: predict(being compared) image path.

		Returns:
			The return float value [0,Positive infinite]. This value represents how many dB of PSNR.
			This more large the value is , the better.

		"""
		img1 = img2np(gt)
		img2 = img2np(predict)
		return PSNR2(img1,img2)

	def calculateSSIM(self, gt, predict):
		"""Calculate SSIM function between two images.

		Args:
			gt: ground truth image path.
			predict: predict(being compared) image path.

		Returns:
			The return float value [0,1]. This value represents how many Similarity of luminance, contrast and structure.
			This more closer to 1 the value is, the better.

		"""		
		ssim_img1 = cv2.imread(gt, 1)
		ssim_img2 = cv2.imread(predict, 1)
		return measure.compare_ssim(ssim_img1, ssim_img2, multichannel=True)

	def calculateMSE(self, gt, predict):
		"""Calculate MSE(Mean-Square Error) function between two images.

		Args:
			gt: ground truth image path.
			predict: predict(being compared) image path.

		Returns:
			The return float value [0,Positive  infinite]. This value represents the Mean-Square Error between the two image.
			This more closer to 0 the value is, the better.

		"""			
		img1 = img2np(gt)
		img2 = img2np(predict)
		return np.mean(np.square(img2 - img1))

	def calculateCosSim(self, gt, predict):
		"""Calculate cosine similarity function between two images.

		Args:
			gt: ground truth image path.
			predict: predict(being compared) image path.

		Returns:
			The return float value [0,1]. This value represents the cosine similarity between the two image.
			This more closer to 0 the value is, the better.

		"""			
		img1 = img2np(gt)
		img2 = img2np(predict)
		img1=np.reshape(img1,(1,img1.shape[1]*img1.shape[2]*img1.shape[3]))
		img2=np.reshape(img2,(1,img2.shape[1]*img2.shape[2]*img2.shape[3]))
		cos_sim = cosine_similarity(img1,img2)
		return cos_sim[0][0]


gt = "/home/c95lpy/image-quality-metrics/output_org20/1.jpg"
predict = "/home/c95lpy/image-quality-metrics/output_blur20/1.jpg"

M = Metric()
result = M.calculateCosSim(gt,predict)
print("cos_sim:", result)
result = M.calculateMSE(gt,predict)
print("MSE:", result)
result = M.calculatePSNR(gt,predict)
print("PSNR:", result)
result = M.calculateSSIM(gt,predict)
print("SSIM:", result)
