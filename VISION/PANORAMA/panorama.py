from os.path import join
import cv2
import imutils
import numpy as np

class Panorama():
	def __init__(self, pano, centrum=180):
		self.katalog_in = '../../OUTPUT/EXAMPLE/'
		self.pano = pano.strip()
		self.centrum = centrum
		self.name_pattern = 'sv_p%s_s640x480_f40_h%s_p%s.png'
		self.isv3 = imutils.is_cv3(or_better=True)

	def set_pano(self, pano):
		self.pano = pano.strip()

	def get_name(self, heading, pitch):
		return self.name_pattern % (self.pano, heading, pitch)

	def get_image(self, name):
		return cv2.imread(join(self.katalog_in, name))

	def read_images(self, width=400):
		self.img_0 = self.get_image(self.get_name('280', '0'))
		self.img_1 = self.get_image(self.get_name('300', '0'))
		self.img_2 = self.get_image(self.get_name('300', '-10'))
		self.img_3 = self.get_image(self.get_name('280', '-10'))
		self.img_4 = self.get_image(self.get_name('260', '-10'))
		self.img_5 = self.get_image(self.get_name('260', '0'))
		self.img_6 = self.get_image(self.get_name('260', '10'))
		self.img_7 = self.get_image(self.get_name('280', '10'))
		self.img_8 = self.get_image(self.get_name('300', '10'))
		self.img_0 = imutils.resize(self.img_0, width=width)
		self.img_1 = imutils.resize(self.img_1, width=width)
		self.img_2 = imutils.resize(self.img_2, width=width)
		self.img_3 = imutils.resize(self.img_3, width=width)
		self.img_4 = imutils.resize(self.img_4, width=width)
		self.img_5 = imutils.resize(self.img_5, width=width)
		self.img_6 = imutils.resize(self.img_6, width=width)
		self.img_7 = imutils.resize(self.img_7, width=width)
		self.img_8 = imutils.resize(self.img_8, width=width)

	def compute_images(self):
		self.p1 = self.compute_two_images(self.img_0, self.img_1)
		self.p2 = self.compute_two_images(self.img_0, self.img_2)
		self.p3 = self.compute_two_images(self.img_0, self.img_3)
		self.p4 = self.compute_two_images(self.img_0, self.img_4)
		self.p5 = self.compute_two_images(self.img_0, self.img_5)
		self.p6 = self.compute_two_images(self.img_0, self.img_6)
		self.p7 = self.compute_two_images(self.img_0, self.img_7)
		self.p8 = self.compute_two_images(self.img_0, self.img_8)

	def show_pano(self):
		self.read_images(width=480)
		self.compute_images()
		self.result = np.zeros((3*self.img_0.shape[0],3*self.img_0.shape[1],3), dtype='uint8')
		self.result[self.img_1.shape[0]+self.p1[1]:2*self.img_1.shape[0]+self.p1[1], self.img_1.shape[1]+self.p1[0]:2*self.img_1.shape[1]+self.p1[0]] = self.img_1		
		self.result[self.img_2.shape[0]+self.p2[1]+0:2*self.img_2.shape[0]+self.p2[1]+0, self.img_2.shape[1]+self.p2[0]:2*self.img_2.shape[1]+self.p2[0]] = self.img_2
		self.result[self.img_3.shape[0]+self.p3[1]:2*self.img_3.shape[0]+self.p3[1], self.img_3.shape[1]+self.p3[0]:2*self.img_3.shape[1]+self.p3[0]] = self.img_3
		self.result[self.img_4.shape[0]+self.p4[1]:2*self.img_4.shape[0]+self.p4[1], self.img_4.shape[1]+self.p4[0]-0:2*self.img_4.shape[1]+self.p4[0]-0] = self.img_4
		self.result[self.img_5.shape[0]+self.p5[1]:2*self.img_5.shape[0]+self.p5[1], self.img_5.shape[1]+self.p5[0]-0:2*self.img_5.shape[1]+self.p5[0]-0] = self.img_5
		self.result[self.img_6.shape[0]+self.p6[1]:2*self.img_6.shape[0]+self.p6[1], self.img_6.shape[1]+self.p6[0]:2*self.img_6.shape[1]+self.p6[0]] = self.img_6
		self.result[self.img_7.shape[0]+self.p7[1]:2*self.img_7.shape[0]+self.p7[1], self.img_7.shape[1]+self.p7[0]:2*self.img_7.shape[1]+self.p7[0]] = self.img_7
		self.result[self.img_8.shape[0]+self.p8[1]:2*self.img_8.shape[0]+self.p8[1], self.img_8.shape[1]+self.p8[0]:2*self.img_8.shape[1]+self.p8[0]] = self.img_8
		self.result[self.img_0.shape[0]:2*self.img_0.shape[0], self.img_0.shape[1]:2*self.img_0.shape[1]] = self.img_0
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
		print(przesX, przesY)
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
	#panorama = Panorama('9cgfAbCfJTS9o0wLjnKc-g')
	panorama = Panorama('FbJIp3vFL_bmrovENrJoRg')
	panorama.show_pano()