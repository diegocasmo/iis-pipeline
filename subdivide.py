import numpy as np
import cv2

def subdivide(image, piecex, piecey, splitx, splity):
	'''  Returns a list containing images. The original image is divided into subimages. 
		 The original image @img is split into a (splitx, splity)-grid of sub-images.
		 Each subimage measures (piecex, piecey).

		 Example:
		 	images = subdivide(img, 100, 100, 2, 2)

		 the image img is subdivided into 2x2 subimages, each measuring 100x100

	'''
	
	# cv2.imshow("Original image", image)
	# cv2.waitKey(0)

	resized_image = cv2.resize(image, (piecex*splitx, piecey*splity))
	
	# cv2.imshow("Resized image", resized_image)
	# cv2.waitKey(0)

	img_array = np.asarray(resized_image)
	images = []	

	i = 0
	while (i<piecey*splity):
		j = 0;
		while (j<piecex*splitx):
			if j < piecex*splitx - piecex + 1 and i < piecey*splity - piecey + 1:
				piece = img_array[i:i+piecey,j:j+piecex]
				images.append(piece)

			#cv2.rectangle(resized_image, (j,i), (j+piecex, i+piecey), (0,0,0), 2)

			j += int(piecex/2)#piecex
		i += int(piecey/2)#piecey


#	cv2.imshow("With grid", resized_image)
#	cv2.waitKey(0)

	return images
'''
im = cv2.imread('cat.jpg', cv2.IMREAD_COLOR)
images = subdivide(im, 32, 32, 8, 8)

for i in images:
	cv2.imshow("im", i)
	cv2.waitKey(0)
'''
