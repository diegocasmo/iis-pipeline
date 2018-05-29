import numpy as np
import cv2
import os as os

pos = [0,0]
def click(event, x, y, flags, param):
	if (event == cv2.EVENT_LBUTTONDOWN):
		pos[0] = x;
		pos[1] = y;
		print ("clicked at x: " + str(x) + ", y: " + str(y) + " and got " + str(pos));

# setup
filenames = ["images/anger/out1.png", "images/anger/out3.png","images/anger/out5.png", "images/anger/out19.png", "images/disgust/out2.png", "images/disgust/out4.png","images/disgust/out15.png", "images/disgust/out16.png", "images/fear/out1.png", "images/fear/out4.png", "images/fear/out5.png", "images/fear/out28.png", "images/happy/out2.png", "images/happy/out4.png", "images/happy/out6.png", "images/happy/out18.png", "images/sadness/out0.png",  "images/sadness/out10.png", "images/sadness/out16.png","images/surprise/out1.png", "images/surprise/out4.png","images/surprise/out8.png", "images/surprise/out12.png"]
features = ["chin_middle.txt", "nchin_middle.txt", "inner_left_eyebrow.txt", "ninner_left_eyebrow.txt", "inner_left_eye_corner.txt", "ninner_left_eye_corner.txt", "inner_right_eyebrow.txt", "ninner_right_eyebrow.txt", "inner_right_eye_corner.txt", "ninner_right_eye_corner.txt", "left_ear_lobe.txt", "nleft_ear_lobe.txt", "left_mouth_corner.txt", "nleft_mouth_corner.txt", "left_nose_peak.txt", "nleft_nose_peak.txt", "left_temple.txt", "nleft_temple.txt", "lower_lip_inner_middle.txt", "nlower_lip_inner_middle.txt", "lower_lip_outer_middle.txt", "nlower_lip_outer_middle.txt", "middle_left_eyebrow.txt", "nmiddle_left_eyebrow.txt", "middle_right_eyebrow.txt", "nmiddle_right_eyebrow.txt", "nose_saddle_left.txt", "nnose_saddle_left.txt", "nose_saddle_right.txt", "nnose_saddle_right.txt", "nose_tip.txt", "nnose_tip.txt", "outer_left_eyebrow.txt", "nouter_left_eyebrow.txt", "outer_left_eye_corner.txt", "nouter_left_eye_corner.txt", "outer_right_eyebrow.txt", "nouter_right_eyebrow.txt", "outer_right_eye_corner.txt", "nouter_right_eye_corner.txt", "right_ear_lobe.txt", "nright_ear_lobe.txt", "right_mouth_corner.txt", "nright_mouth_corner.txt", "right_nose_peak.txt", "nright_nose_peak.txt", "right_temple.txt", "nright_temple.txt", "upper_lip_outer_middle.txt", "nupper_lip_outer_middle.txt"]

for feature in features:
	with open(feature, "w") as outfile:
		outfile.write("# Generated with script data_label.py\n")

cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("face", click)
counter = 1

# show every image
for filename in filenames:
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	image = cv2.resize(image, (512,512))
	#indicate each feature
	for feature in features:
		print("Feature " + str(feature) + ":")
		cv2.imshow("face", image)
		key = cv2.waitKey(0)
		if key == ord('c'):
			continue
		subimage = image[pos[1]-8:pos[1]+8, pos[0]-8:pos[0]+8]
		outname = "./images2/" + str(counter) + ".png"
		counter += 1
		cv2.imwrite(outname, subimage, [int(cv2.IMWRITE_PNG_COMPRESSION)])
		with open(feature, "a") as outfile:
			outfile.write(outname + "\n")
