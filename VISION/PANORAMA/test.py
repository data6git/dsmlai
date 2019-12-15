'''
	GOAL - split two images into 7x7 parts each and split all parts in one panorama. Get center and upper right images for test
'''
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

	def compute_images(self, center):
		self.params = []
		self.images_pano = []
		self.wykorzystane = dict()
		for idx in range(len(self.images)):
		#for idx in range(5):
			self.wykorzystane[idx] = []
			for i in range(0,len(self.images)):
				przes = self.compute_two_images(self.images[idx], self.images[i])
				if przes is not None:
					self.params.append(przes)
					self.images_pano.append(self.images[i])
					if i != idx:
						self.wykorzystane[idx].append(i)
		print('-'*80)
		for klucz in self.wykorzystane:
			print(klucz, self.wykorzystane[klucz])
		print('-'*80)

	def compute_two_images(self, imageA, imageB, ratio=0.75, reprojTresh=1.0, showMatches=False):
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		if len(kpsA) == 0 or len(kpsB) == 0:
			return None

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
		if len(shift_X)>20 and len(shift_Y)>20:
			#print(len(shift_X), len(shift_Y))
			przesX = int(sum(shift_X)/len(shift_X))
			przesY = int(sum(shift_Y)/len(shift_Y))
			return (przesX, przesY)
		else:
			return None

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

	def split_image(self, image):
		parts = []
		shape = image.shape
		shape_X = shape[0]
		shape_Y = shape[1]
		ratio_X = shape_X/3
		ratio_Y = shape_Y/3
		counter = 0
		for x in range(5):
			X_str = int(x*ratio_X/2)
			X_stp = int(X_str+ratio_X)
			#print(X_str, X_stp)
			for y in range(5):
				Y_str = int(y*ratio_Y/2)
				Y_stp = int(Y_str+ratio_Y)
				#print(image.shape, X_str, X_stp, Y_str, Y_stp)
				part = image[X_str:X_stp, Y_str:Y_stp]
				#cv2.imshow('image%d'%counter, part)
				counter += 1
				parts.append(part)
		return parts

	def read_images(self, width=400, str_X=180):
		self.images = []
		img_params = [(str_X, 0), (str_X-20, 10), (str_X+20, 10)]
		for param in img_params:
			image = self.get_image(self.get_name(param[0], param[1]))
			image = imutils.resize(image, width=width)
			parts = self.split_image(image)
			for part in parts:
				self.images.append(part)

	def create_tree_based_on_dict(self):
		#tworzymy drzewo w oparciu o dane ze slownika
		self.zbior_wszystkich_elementow = set()
		for element in range(len(self.params)):
			self.zbior_wszystkich_elementow.add(element)
		self.pozycje = dict()
		self.pozycje[0] = [0, 0]
		for klucz1 in self.wykorzystane:
			obr1 = self.images[klucz1]
			for klucz2 in self.wykorzystane[klucz1]:
				obr2 = self.images[klucz2]
				wsp = self.compute_two_images(obr1, obr2)
				print(klucz1, klucz2, wsp)

	def show_pano(self, width=400, str_X=180):
		self.read_images(width=width, str_X=str_X)
		self.compute_images(str_X)
		#tworzymy zbior wszystkich kawalkow wszystkich obrazkow
		#iterujemy po self.wykorzystane i zapisujemy wspolrzedne relatywne czyli do klucza
		#startem jest klocek o id=0
		#self.create_tree_based_on_dict()
		self.pozycje = dict()
		self.pozycje[0] = [0, 0]
		for klucz1 in self.wykorzystane:
			obr1 = self.images[klucz1]
			for klucz2 in self.wykorzystane[klucz1]:
				if klucz2 not in self.pozycje and klucz1 in self.pozycje:
					obr2 = self.images[klucz2]
					wsp = self.compute_two_images(obr1, obr2)
					if wsp is not None:
						#print(klucz1, klucz2, wsp[0], wsp[1])
						wsp_ref = self.pozycje[klucz1]
						wsp_dst = [0, 0]
						wsp_dst[0] = wsp_ref[0]+wsp[0]
						wsp_dst[1] = wsp_ref[1]+wsp[1]
						self.pozycje[klucz2] = wsp_dst
		for klucz in sorted(self.pozycje):
			print(klucz, self.pozycje[klucz])
		'''
		for klucz in self.wykorzystane:
			print(klucz, self.wykorzystane[klucz])
		for i in range(len(self.params)):
			print(i, self.params[i],self.images_pano[i].shape[0], self.images_pano[i].shape[1])
		'''
		self.result = np.zeros((7*self.images_pano[0].shape[0],7*self.images_pano[0].shape[1],3), dtype='uint8')
		for i in range(len(self.pozycje)): #bylo params
			#self.result[self.images[i].shape[0]+self.params[i][1]:2*self.images[i].shape[0]+self.params[i][1], self.images[i].shape[1]+self.params[i][0]:2*self.images[i].shape[1]+self.params[i][0]] = self.images[i]
			#cv2.imshow('Image%s'%i, self.images_pano[i])
			try:
				#str_x = 3*self.images_pano[i].shape[0] + 1*self.pozycje[i][1]
				#stp_x = 4*self.images_pano[i].shape[0] + 1*self.pozycje[i][1]
				#str_y = 3*self.images_pano[i].shape[1] + 1*self.pozycje[i][0]
				#stp_y = 4*self.images_pano[i].shape[1] + 1*self.pozycje[i][0]
				str_x = 3*self.images[i].shape[0] + 1*self.pozycje[i][1]
				stp_x = 4*self.images[i].shape[0] + 1*self.pozycje[i][1]
				str_y = 3*self.images[i].shape[1] + 1*self.pozycje[i][0]
				stp_y = 4*self.images[i].shape[1] + 1*self.pozycje[i][0]
				self.result[str_x:stp_x, str_y:stp_y] = self.images[i]
				print(i, self.pozycje[i],self.images_pano[i].shape[0], self.images_pano[i].shape[1], str_x, stp_x, str_y, stp_y)
			except:
				print(i, 'ERROR')
		cv2.imshow('Panorama', self.result)
		cv2.waitKey(0)
		

if __name__ == '__main__':
	panorama = Panorama('9cgfAbCfJTS9o0wLjnKc-g')
	panorama.show_pano(640,200)