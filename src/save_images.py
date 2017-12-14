#!/usr/bin/env python

import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required = True, help = "Path to image dataset")
ap.add_argument("-l","--labels",required = True, help = "Path to image labels")
ap.add_argument("-o","--outcsv",required = True, help = "Path to output csv file")
ap.add_argument("-j","--jpgimg",required = True, help = "Path to jpg images")
ap.add_argument("-n","--num",required = True, help = "No. of images")
args = vars(ap.parse_args())

f = open(args["images"],"rb")
l = open(args["labels"],"rb")
csv = open(args["outcsv"],"w")

f.read(16)	#First 16 bytes not pixels
l.read(8)	#First 8 bytes not labels

images = []

# csv file ==> LABEL(1) | IMAGE(784)
# 1 byte per label
# 28x28 = 784 bytes pixels per image

for i in range(int(args["num"])):
	image = [ord(l.read(1))]
	for j in range(784):
		image.append(ord(f.read(1)))
	img = np.asarray([[image[1+p*28+q] for q in range(28)] for p in range(28)],dtype = np.float32)
	cv2.imwrite(args["jpgimg"]+"/im_"+str(i).zfill(5)+".jpg",img)
	images.append(image)

for image in images:
	csv.write(",".join(str(pixel) for pixel in image)+"\n")

f.close()
l.close()
csv.close()