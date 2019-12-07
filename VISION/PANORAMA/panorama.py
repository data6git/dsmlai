from os.path import join
import cv2
import imutils
import numpy as np

class Panorama():
	def __init__(self, pano, centrum=180):
		self.katalog_in = 'EXAMPLES/'
		self.pano = pano.strip()
		self.centrum = centrum
		self.name_pattern = 'sv_p%s_s640x480_f40_h%s_p%s.png'
		self.isv3 = imutils.is_cv3(or_better=True)

	def set_pano(self, pano):
		self.pano = pano.strip()

	def get_name(self, heading, pitch):
		heading = heading % 360
		heading_s = '%d' % heading
		while len(heading_s)<3:
			heading_s = '0'+heading_s
		return self.name_pattern % (self.pano, heading_s, pitch)

	def get_image(self, name):
		return cv2.imread(join(self.katalog_in, name))

	def read_images(self, width=400, str_X=180):
		self.images = []
		img_params = [(str_X, 0), (str_X+20, 0), (str_X+20, -10), (str_X, -10), (str_X-20, -10), (str_X-20, 0), (str_X-20, 10), (str_X, 10), (str_X+20, 10)]
		for param in img_params:
			self.images.append(self.get_image(self.get_name(param[0], param[1])))
		for i in range(len(self.images)):
			self.images[i] = imutils.resize(self.images[i], width=width)

	def compute_images(self):
		self.params = []
		for i in range(0,len(self.images)):
			self.params.append(self.compute_two_images(self.images[0], self.images[i]))

	def show_pano(self, width=400, str_X=180):
		self.read_images(width=width, str_X=str_X)
		self.compute_images()
		self.result = np.zeros((3*self.images[0].shape[0],3*self.images[0].shape[1],3), dtype='uint8')
		#for i in range(1,len(self.params)):
		for i in (2,4,6,8,1,3,5,7,0):
			self.result[self.images[i].shape[0]+self.params[i][1]:2*self.images[i].shape[0]+self.params[i][1], self.images[i].shape[1]+self.params[i][0]:2*self.images[i].shape[1]+self.params[i][0]] = self.images[i]
		#for i in range(1):
		#	self.result[self.images[i].shape[0]+self.params[i][1]:2*self.images[i].shape[0]+self.params[i][1], self.images[i].shape[1]+self.params[i][0]:2*self.images[i].shape[1]+self.params[i][0]] = self.images[i]
		cv2.imshow('Panorama', self.result)
		cv2.waitKey(0)

	def compute_two_images(self, imageA, imageB, ratio=0.75, reprojTresh=4.0, showMatches=False):
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojTresh)
		if M is None:
			return None
		(matches, H, status) = M
		shift_X = []
		shift_Y = []
		for match, sts in zip(matches, status):
			if sts == 1:
				shift_X.append(kpsA[match[1]][0]-kpsB[match[0]][0])
				shift_Y.append(kpsA[match[1]][1]-kpsB[match[0]][1])				
		przesX = int(sum(shift_X)/len(shift_X))
		przesY = int(sum(shift_Y)/len(shift_Y))
		return (przesX, przesY)

	def detectAndDescribe(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if self.isv3:
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
		else:
			detector = cv2.FeatureDetector_create('SIFT')
			kps = detector.detect(gray)
			extractor = cv2.DescriptorExtractor_create('SIFT')
			(kps, features) = extractor.compute(gray, kps)

		kps = np.float32([kp.pt for kp in kps])

		return(kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojTresh):
		matcher = cv2.DescriptorMatcher_create('BruteForce')
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		for m in rawMatches:
			if len(m)==2 and m[0].distance<m[1].distance*ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		if len(matches) > 4:
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojTresh)

			return (matches, H, status)

		else:
			return None


if __name__ == '__main__':
	panorama = Panorama('CAoSLEFGMVFpcE81VElidWJkbUhsQzJHZkpZSml6WEExNkhHRTNLbXR3YkdqRGU5')
	panorama.show_pano(400,120)