# import required module
import os
import sys
import numpy as np
import pyopencl

from PIL import Image
from SSIM_PIL import compare_ssim
from scipy.stats import wasserstein_distance

frames = []
images = []
avg_pixel_dif = []
avg_frame_dif = []
earth_mover_dist = []
struct_sim = []


def create_histogram(img):
  h, w = img.shape
  hist = [0.0] * (h * w)
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)
   

def calculate_earth_movers_distance(img_a, img_b):
  hist_a = create_histogram(img_a)
  hist_b = create_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)

def structural_sim(img_a, img_b):
  sim = compare_ssim(img_a, img_b)
  return sim


if __name__ == '__main__':
    frames = []
    avg_pixel_dif = []
    avg_frame_dif = []
    earth_mover_dist = []

    if(len(sys.argv) < 1):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 Detect_Cut.py 'a path' and a threshold")

    #grab the grid size
    directory = sys.argv[1]

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            img = img.convert("L")
            images.append(img)
            img = np.array(img)
            frames.append(img)
    
    for i in range(len(frames)):
        if(i < (len(frames)-1)):
            img1 = frames[i]
            img2 = frames[i + 1]

            difference = 0

            avgInt1 = 0
            avgInt2 = 0

            for row in range(len(img1)):
                for col in range(len(img1[row])):
                    p1 = img1[row][col]
                    p2 = img2[row][col]
                    difference += abs(p1 - p2)

                    avgInt1 += p1
                    avgInt2 += p2
            
            difference = difference / (len(img1) * len(img1[1]))
            avgInt1 = avgInt1 / (len(img1) * len(img1[1]))
            avgInt2 = avgInt2 / (len(img1) * len(img1[1]))

            #print("Frame # " + str(i + 1) + " : difference = " + str(difference) + " : avgInt = " + str(abs(avgInt1 - avgInt2)) + " : earthMover = " + str(calculate_earth_movers_distance(img1, img2)))

            avg_pixel_dif.append(difference)
            avg_frame_dif.append(abs(avgInt1 - avgInt2))
            earth_mover_dist.append(calculate_earth_movers_distance(img1, img2))
            struct_sim.append(structural_sim(images[i], images[i+1]))
    
    print("Best pixel dif = " + str(max(avg_pixel_dif)) + " at frame " + str(avg_pixel_dif.index(max(avg_pixel_dif)) + 1) + " and " + str(avg_pixel_dif.index(max(avg_pixel_dif)) + 2))
    print("Best frame dif = " + str(max(avg_frame_dif)) + " at frame " + str(avg_frame_dif.index(max(avg_frame_dif)) + 1) + " and " + str(avg_frame_dif.index(max(avg_frame_dif)) + 2))
    print("Best earthmover dif = " + str(max(earth_mover_dist)) + " at frame " + str(earth_mover_dist.index(max(earth_mover_dist)) + 1) + " and " + str(earth_mover_dist.index(max(earth_mover_dist)) + 2))
    print("Best structsim dif = " + str(max(struct_sim)) + " at frame " + str(struct_sim.index(min(struct_sim)) + 1) + " and " + str(struct_sim.index(min(struct_sim)) + 2))