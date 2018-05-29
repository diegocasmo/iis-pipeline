import pyrealsense2 as rs
import numpy as np
import cv2
import subdivide as sd
from scipy.misc import imresize
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def read_file(filename, dir_path = "./"):
	"""
	Returns flattened images from files with filenames
	contained in the file with passed filename

	Args:
		filename(str): Name of file containing filenames of images

	Returns:
		list of numpy.ndarray: The images indicated by the contents of filename, flattened
	"""

	infile = open(filename)

	outdata = []
	for line in infile:
		line = line[:-1]
		if len(line) > 0:
			if line[0] == '#': # lines starting with '#' are comments
				continue
			line = dir_path + line
			im = cv2.imread(line, cv2.IMREAD_COLOR)
			if im is not None:
				#convert to grayscale
				im = np.asanyarray(im).astype(np.float32)
				im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				im -= np.min(im[:])
				im /= np.max(im[:])
				im = im * 255
				im = im.astype(np.uint8)

				im.flatten()
				outdata.append(im)
			else:
				print("Bad input line: " + str(line) + " in file " + filename + " ")
	infile.close()
	return outdata

def balance_classes(class1, class2):
	"""
	Balances two lists to be of the same length

	Args:
		class1(list): First of the lists to balance
		class2(list): Second of the lists to balance	

	Returns:
		list: The first len(class2) elements of class1
		list: The first len(class1) elements of class2
	"""

	n = min(len(class1), len(class2))
	class1 = class1[:n]
	class2 = class2[:n]
	return class1, class2

def holdout(class1, class2, split):
	"""
	Generates training data, training labels, test data, and test labels from
	the passed classes. Labels positive samples with '0', and negative samples with '1'

	Args:
		class1(list): Positive samples
		class2(list): Negative samples
		split(float): Percent of samples to use for training

	Returns:
		list: Samples for training
		list of int: Labels for training
		list: Samples for testing
		list of int: Labels for testing
	"""

	training_data = class1[:int(split*len(class1))] + class2[:int(split*len(class2))]
	training_labels = [0]*int((split*len(class1))) + [1]*int((split*len(class2)))
	test_data = class1[int(split*len(class1)):] + class2[int(split*len(class2)):]
	test_labels = [0]*(len(class1)-int(split*len(class1))) + [1]*(len(class2)-int(split*len(class2)))
	return (training_data, training_labels, test_data, test_labels)

def train_classifier(featdata, nfeatdata, split, featname = "unnamed feature", testclassifier = True):
	"""
	Creates, trains and tests an SVM classifier based on passed data

	Side-effects:
		Prints to stdout the accuracy score and feat name of the classifier.

	Args:
		featdata(list): Positive samples
		nfeatdata(list): Negative samples
		split(float): Percent of samples to use for training
		featname(str): Name of the feat to classify
		testclassifier(bool): False if the classifier need not be tested. Note that
							  the value of `split` is still taken into account
	Returns:
		LinearSVC: SVM classifier based on passed data
	"""

	fdata, ndata = balance_classes(featdata, nfeatdata)
	training_data, training_labels, test_data, expected_labels = holdout(fdata, ndata, split)
	training_data = np.asarray(training_data)
	training_labels = np.asarray(training_labels)
	test_data = np.asarray(test_data)
	expected_labels = np.asarray(expected_labels)
	training_data = training_data.reshape((len(training_data) ,-1))
	if testclassifier:
		test_data = test_data.reshape((len(test_data), -1))

	clf_svm = LinearSVC(tol = 0.000001)
	clf_svm.fit(training_data, training_labels)

	if testclassifier:
		predicted_labels = clf_svm.predict(test_data)
		acc_svm = accuracy_score(expected_labels, predicted_labels)
		print("Linear SVM accuracy for " + featname + ": " + str(acc_svm))
	return clf_svm

def closest_point (image):
	"""
	Finds the closest point of the passed image, given that the passed
	image is a depth image

	Args:
		image(numpy.ndarray): Depth image of which to find the closest point

	Returns:
		int: X-coordinate of the closest point
		int: Y-coordinate of the closest point
	"""

	x, y = np.unravel_index(np.argmin(np.ma.masked_equal(image, 0.0, copy=False), axis=None), image.shape)
	x = max(x, 1)
	x = min(x, 511)
	y = max(y, 1)
	y = min(y, 511)
	return x, y

class FeatureReader:
	"""
	Class for an object which reads features from a Realsense camera
	To use, create an object, and then call frame() when a new frame is to be read.
	To display the read image, use the displayimage() method.
	Also contains some utility methods. 
	"""


	def __init__ (self, dir_path = "./"):

		# a black background image
		#image_size = (1280,720)
		self.image_size = (640,480)
		self.background = np.zeros((self.image_size[1],self.image_size[0],3))

		# the openCV window
		self.windowname = 'Features'
		cv2.namedWindow(self.windowname, cv2.WINDOW_AUTOSIZE)

		# the trained haar-cascade classifier data
		self.face_cascade = cv2.CascadeClassifier(dir_path + 'frontal_face_features.xml')

		# configure the realsense camera
		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, self.image_size[0], self.image_size[1], rs.format.bgr8, 30)
		config.enable_stream(rs.stream.depth, self.image_size[0], self.image_size[1], rs.format.z16, 30)
		self.frame_aligner = rs.align(rs.stream.color)

		# Start streaming
		profile = self.pipeline.start(config)
		depth_sensor = profile.get_device().first_depth_sensor()
		self.depth_scale = depth_sensor.get_depth_scale()

		# Add file names of the feat files and non-feat files here
		ffilenames = ["chin_middle.txt", "inner_left_eyebrow.txt", "inner_left_eye_corner.txt", "inner_right_eyebrow.txt", "inner_right_eye_corner.txt", "left_ear_lobe.txt", "left_mouth_corner.txt", "left_nose_peak.txt", "left_temple.txt", "lower_lip_inner_middle.txt", "lower_lip_outer_middle.txt", "middle_left_eyebrow.txt", "middle_right_eyebrow.txt", "nose_saddle_left.txt", "nose_saddle_right.txt", "nose_tip.txt", "outer_left_eyebrow.txt", "outer_left_eye_corner.txt", "outer_right_eyebrow.txt", "outer_right_eye_corner.txt", "right_ear_lobe.txt", "right_mouth_corner.txt", "right_nose_peak.txt", "right_temple.txt", "upper_lip_outer_middle.txt"]
		nfilenames = ["nchin_middle.txt", "ninner_left_eyebrow.txt", "ninner_left_eye_corner.txt", "ninner_right_eyebrow.txt", "ninner_right_eye_corner.txt", "nleft_ear_lobe.txt", "nleft_mouth_corner.txt", "nleft_nose_peak.txt", "nleft_temple.txt", "nlower_lip_inner_middle.txt", "nlower_lip_outer_middle.txt", "nmiddle_left_eyebrow.txt", "nmiddle_right_eyebrow.txt", "nnose_saddle_left.txt", "nnose_saddle_right.txt", "nnose_tip.txt", "nouter_left_eyebrow.txt", "nouter_left_eye_corner.txt", "nouter_right_eyebrow.txt", "nouter_right_eye_corner.txt", "nright_ear_lobe.txt", "nright_mouth_corner.txt", "nright_nose_peak.txt", "nright_temple.txt", "nupper_lip_outer_middle.txt"]

		for i in range(len(ffilenames)):
			ffilenames[i] = dir_path + ffilenames[i]
		for i in range(len(nfilenames)):
			nfilenames[i] = dir_path + nfilenames[i]

		self.fclassifiers = []
		self.nfeats = len(ffilenames)
		self.history = np.empty((1,self.nfeats,3))

		# for each feature:
		for i in range(len(ffilenames)):
			if i == 0:
				self.fclassifiers.append(float('nan'))
				continue #TODO skipping chin_middle
			# Read training data
			fdata = read_file(ffilenames[i], dir_path)
			ndata = read_file(nfilenames[i], dir_path)
			# Train classifier
			self.fclassifiers.append(train_classifier(fdata, ndata, 1, ffilenames[i], testclassifier=False))

	def frame (self):
		# Wait for a new frame and align the frame
		frames = self.pipeline.wait_for_frames()
		aligned_frames = self.frame_aligner.process(frames)
		depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()
		if not depth_frame or not color_frame:
			raise RuntimeError('Frame skipped')
		# image to display
		self.image = np.asanyarray(color_frame.get_data()).astype(np.float32)
		self.image -= np.min(self.image[:])
		self.image /= np.max(self.image[:])
		#convert to grayscale
		gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		gray_image -= np.min(gray_image[:])
		gray_image /= np.max(gray_image[:])
		gray_image_uint = gray_image * 255
		gray_image_uint = gray_image_uint.astype(np.uint8)

		# the face detection
		mask = np.ones((self.image.shape[0],self.image.shape[1]))
		faces = self.face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)
		depth_image = np.asanyarray(depth_frame.get_data())

		outdata = [[0]*3]*self.nfeats

		# process faces
		if len(faces) > 0:
			for (x,y,w,h) in faces:
				# draw a rectangle where a face is detected
				cv2.rectangle(self.image, (x,y),(x+w,y+h), (255,0,0), 2)
				# create a rescaled copy of the rectangle, and the corresponding depth rectangle
				im = cv2.resize(gray_image_uint[y:y+h, x:x+w], (512,512))
				imd = cv2.resize(depth_image[y:y+h, x:x+w], (512,512))
				# find point closest to camera, since that should be the nose
				nose_index_x, nose_index_y = closest_point(imd)

				#TODO: this assumes that the midpoint of the bounding box is the nose
				nose_index_x, nose_index_y = (256, 256)

				quad_width_x, quad_width_y = (16,16)
				quad_freq_x, quad_freq_y = (16,16)

				# upper left quadrant
				# create smaller rectangles based on the large rectangles, with overlap
				classifier_indices = [1, 2, 8, 11, 13, 16, 17] # which classifiers are relevant for the quadrant?
				chunks = sd.subdivide(im[:nose_index_y, :nose_index_x], quad_width_x, quad_width_y, quad_freq_x, quad_freq_y)
				chunks = np.asarray(chunks)
				chunks = chunks.reshape((len(chunks) ,-1))
				for i in classifier_indices:
					if i == 0:
						continue #TODO skipping chin_middle
					confidence_scores = self.fclassifiers[i].decision_function(chunks)
					pos = np.argmin(confidence_scores)
					x_quadrant, y_quadrant = np.unravel_index(pos, (2*quad_freq_x-1, 2*quad_freq_y-1))
					x_quadrant = (quad_width_x/2) * (x_quadrant+1)
					y_quadrant = (quad_width_y/2) * (y_quadrant+1)
					scale_x, scale_y = ((nose_index_x)/(quad_freq_x*quad_width_x), (nose_index_y)/(quad_freq_y*quad_width_y))
					x_im, y_im = (int((x_quadrant) * scale_x), int((y_quadrant) * scale_y))
					scale_x, scale_y = (w/512.0, h/512.0)
					xpos, ypos = (int(x_im * scale_x) + x, int(y_im * scale_y) + y)
					zpos = depth_image[ypos,xpos]
					outdata[i] = [xpos,ypos,zpos]

				# upper right quadrant
				# create smaller rectangles based on the large rectangles, with overlap
				classifier_indices = [3, 4, 12, 14, 18, 19, 23] # which classifiers are relevant for the quadrant?
				chunks = sd.subdivide(im[:nose_index_y, nose_index_x:], quad_width_x, quad_width_y, quad_freq_x, quad_freq_y)
				chunks = np.asarray(chunks)
				chunks = chunks.reshape((len(chunks) ,-1))
				for i in classifier_indices:
					if i == 0:
						continue #TODO skipping chin_middle
					confidence_scores = self.fclassifiers[i].decision_function(chunks)
					pos = np.argmin(confidence_scores)
					x_quadrant, y_quadrant = np.unravel_index(pos, (2*quad_freq_x-1, 2*quad_freq_y-1))
					x_quadrant = (quad_width_x/2) * (x_quadrant+1)
					y_quadrant = (quad_width_y/2) * (y_quadrant+1)
					scale_x, scale_y = ((512.0-nose_index_x)/(quad_freq_x*quad_width_x), (nose_index_y)/(quad_freq_y*quad_width_y))
					x_im, y_im = (int((x_quadrant) * scale_x) + nose_index_x, int((y_quadrant) * scale_y))
					scale_x, scale_y = (w/512.0, h/512.0)
					xpos, ypos = (int(x_im * scale_x) + x, int(y_im * scale_y) + y)
					zpos = depth_image[ypos,xpos]
					outdata[i] = [xpos,ypos,zpos]

				# lower left quadrant
				# create smaller rectangles based on the large rectangles, with overlap
				classifier_indices = [0, 5, 6, 7, 15, 24] # which classifiers are relevant for the quadrant?
				chunks = sd.subdivide(im[nose_index_y:, :nose_index_x], quad_width_x, quad_width_y, quad_freq_x, quad_freq_y)
				chunks = np.asarray(chunks)
				chunks = chunks.reshape((len(chunks) ,-1))
				for i in classifier_indices:
					if i == 0:
						continue #TODO skipping chin_middle
					confidence_scores = self.fclassifiers[i].decision_function(chunks)
					pos = np.argmin(confidence_scores)
					x_quadrant, y_quadrant = np.unravel_index(pos, (2*quad_freq_x-1, 2*quad_freq_y-1))
					x_quadrant = (quad_width_x/2) * (x_quadrant+1)
					y_quadrant = (quad_width_y/2) * (y_quadrant+1)
					scale_x, scale_y = ((nose_index_x)/(quad_freq_x*quad_width_x), (512.0-nose_index_y)/(quad_freq_y*quad_width_y))
					x_im, y_im = (int((x_quadrant) * scale_x), int((y_quadrant) * scale_y) + nose_index_y)
					scale_x, scale_y = (w/512.0, h/512.0)
					xpos, ypos = (int(x_im * scale_x) + x, int(y_im * scale_y) + y)
					zpos = depth_image[ypos,xpos]
					outdata[i] = [xpos,ypos,zpos]

				# lower right quadrant
				# create smaller rectangles based on the large rectangles, with overlap
				classifier_indices = [9, 10, 20, 21, 22] # which classifiers are relevant for the quadrant?
				chunks = sd.subdivide(im[nose_index_y:, nose_index_x:], quad_width_x, quad_width_y, quad_freq_x, quad_freq_y)
				chunks = np.asarray(chunks)
				chunks = chunks.reshape((len(chunks) ,-1))
				for i in classifier_indices:
					if i == 0:
						continue #TODO skipping chin_middle
					confidence_scores = self.fclassifiers[i].decision_function(chunks)
					pos = np.argmin(confidence_scores)
					x_quadrant, y_quadrant = np.unravel_index(pos, (2*quad_freq_x-1, 2*quad_freq_y-1))
					x_quadrant = (quad_width_x/2) * (x_quadrant+1)
					y_quadrant = (quad_width_y/2) * (y_quadrant+1)
					scale_x, scale_y = ((512.0-nose_index_x)/(quad_freq_x*quad_width_x), (512.0-nose_index_y)/(quad_freq_y*quad_width_y))
					x_im, y_im = (int((x_quadrant) * scale_x) + nose_index_x, int((y_quadrant) * scale_y) + nose_index_y)
					scale_x, scale_y = (w/512.0, h/512.0)
					xpos, ypos = (int(x_im * scale_x) + x, int(y_im * scale_y) + y)
					zpos = depth_image[ypos,xpos]
					outdata[i] = [xpos,ypos,zpos]
				outdata[0] = [float('nan'), float('nan'), float('nan')]
				
		# Create mask
		mask = np.ones((self.image.shape[0],self.image.shape[1]))
		# alpha blending
		self.image = mask[...,None] * self.image + (1 - mask[...,None]) * self.background

		outdata = np.asarray(outdata)
		return outdata

	def dots (self, arr):
		"""
		Paints dots on the current image, in positions given as arguments.
		This method should be called before the displayimage method

		Args:
			arr(list of list of ints): The positions of the dots to paint
		"""
		
		arr = np.asarray(arr)
		for i in range(len(arr)):
			if np.isnan(arr[i][0]) or np.isnan(arr[i][1]):
				continue
			if i == 2:
				cv2.circle(self.image, (int(round(arr[i][0])),int(round(arr[i][1]))), 2, (0,0,255), -1)
				continue
			cv2.circle(self.image, (int(round(arr[i][0])),int(round(arr[i][1]))), 2, (0,255,0), -1)

	def displayimage(self):
		"""
		Displays the current image.
		"""

		# Show images
		cv2.imshow(self.windowname, self.image)
		cv2.waitKey(1)

	def avrg_frames(self,pos,depth=5):
		"""
		Returns the average feature positions of recent frames

		Args:
			pos(list of list of int): List of the most recent feature positions
			depth(int): The number of previous iterations to remember

		Returns:
			list of list of int: List of the average feature positions
		"""

		if len(self.history) < depth:
			outpos = np.expand_dims(pos, 0)
			self.history = np.concatenate((self.history, outpos), 0)
			return pos

		if len(self.history) >= depth:
			self.history = np.delete(self.history, 0, 0)
		pos = np.expand_dims(pos, 0)
		self.history = np.concatenate((self.history, pos), 0)

		outdata = [list(map(lambda i: np.nanmean(self.history[:,i,0]), range(self.nfeats))), list(map(lambda i: np.nanmean(self.history[:,i,1]), range(self.nfeats))), list(map(lambda i: np.nanmean(self.history[:,i,2]), range(self.nfeats)))]
		outdata = np.transpose(outdata)
		return outdata

	def finalize (self):
		"""
		Stops the feature reader in a nice way
		"""
		# Stop streaming
		self.pipeline.stop()
