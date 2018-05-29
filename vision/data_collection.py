import pyrealsense2 as rs
import numpy as np
import cv2
import time
import subdivide as sd

# a black background image
#image_size = (1280,720)
image_size = (640,480)
background = np.zeros((image_size[1],image_size[0],3))

# the openCV window
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# the trained haar-cascade classifier data
face_cascade = cv2.CascadeClassifier('frontal_face_features.xml')

# configure the realsense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, image_size[0], image_size[1], rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, image_size[0], image_size[1], rs.format.z16, 30)
frame_aligner = rs.align(rs.stream.color)

# Start streaming
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Start timing
start_time = time.time()
ref_time = start_time

# For output
out_counter = 0
out_file = "./images/out"
save_frequency = 1

try:
	while True:
		time_elapsed = time.time() - ref_time
		# Wait for a new frame and align the frame
		frames = pipeline.wait_for_frames()
		aligned_frames = frame_aligner.process(frames)
		depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()

		if not depth_frame or not color_frame:
			continue

		# image to display
		image = np.asanyarray(color_frame.get_data()).astype(np.float32)
		image -= np.min(image[:])
		image /= np.max(image[:])

		#convert to grayscale
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image -= np.min(gray_image[:])
		gray_image /= np.max(gray_image[:])
		gray_image_uint = gray_image * 255
		gray_image_uint = gray_image_uint.astype(np.uint8)

		# the face detection
		mask = np.ones((image.shape[0],image.shape[1]))
		faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)
		depth_image = np.asanyarray(depth_frame.get_data())
		if len(faces) > 0:
			for (x,y,w,h) in faces:
				# draw a rectangle where a face is detected
				cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,0), 2)

		# alpha blending
		image = mask[...,None] * image + (1 - mask[...,None]) * background

		# Show images
		cv2.imshow('RealSense', image)
		cv2.waitKey(1)

		if time_elapsed > save_frequency:
			image = image * 255.0
			# Print entire image
			filename = out_file + str(out_counter) + ".png"
			cv2.imwrite(filename, image, [int(cv2.IMWRITE_PNG_COMPRESSION)])
			# Update/reset
			ref_time = time.time()
			out_counter += 1
			
except KeyboardInterrupt:
    print("\nShutting down -- Good Bye")
finally:
    # Stop streaming
    pipeline.stop()


