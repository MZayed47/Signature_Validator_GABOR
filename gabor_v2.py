import os
from zipfile import ZIP_BZIP2
import cv2 as cv
import numpy as np
from math import pi
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# **Defines**

# %%
N1, N2 = 5, 8
h, w = int(60/N1), int(120/N2)


def normalize(image):
  img = image.copy()
  return (img - np.mean(img))/np.std(img)

def GT(img, kernel):
  out = cv.filter2D(img, -1, kernel/np.sum(kernel))
  return normalize(out).copy()

def Box(img, i, j, h, w):
  return img[i*h:i*h+h, j*w:j*w+w].copy()

def findF(img):
  Fs = []
  for i in range(N1):
    for j in range(N2):
      box = Box(img, i, j, h, w)
      F = 0
      n = 0
      for x_i in range(box.shape[0]):
        for y_j in range(box.shape[1]):
          if(box[x_i][y_j]>0.5):
            n+=1
            F += np.arctan((h-x_i+1)/(y_j+1))/np.pi*180
      if(n!=0):
        F = F / n
      Fs += [F]
  return Fs.copy()

def findF_for_all_kernels(img, kernels):
  Fs = []
  for kernel in kernels:
    out = GT(img, kernel)
    Fs += findF(out)
  return np.array(Fs).copy()

def findF_for_all_data(genuines, skills, kernels):
  Fs = []
  for i in range(len(genuines)):
    Fs += [findF_for_all_kernels(genuines[i], kernels)]
  for i in range(len(skills)):
    Fs += [findF_for_all_kernels(skills[i], kernels)]
  Fs = np.array(Fs)
  Fs = StandardScaler().fit_transform(Fs)
  Fs = PCA(3).fit_transform(Fs)
  return Fs[:len(genuines)], Fs[len(genuines):]

def findMuSigma(Fs, kernels):
  mu = np.sum(Fs, axis=0)/len(Fs)
  Fs_mu = Fs-mu
  sigma = Fs_mu.T@Fs_mu/len(Fs)
  return mu, sigma

def findD(Fs, mu, sigma):
  d = []
  for i in range(len(Fs)):
    d += [np.sqrt(np.array(Fs[i]-mu)@np.linalg.inv(sigma)@np.transpose([Fs[i]-mu]))]
  return np.array(d).T[0]


def GT_similarity(path1, path2):
    kernels = []
    for u in ([0, 2, 4, 6]):
        for v in [1, 2, 3]:
            params = {'ksize': (7,7), 'sigma': 3, 'theta': u*pi/8, 'lambd': 7/v, 'gamma': 1, 'psi' : 0}
            kernels += [cv.getGaborKernel(**params, ktype = cv.CV_32F)]

    ss = 0
    for i in range(4):
      rr = './temp/' + path1 + str(i+1) + '.jpg'

      img_a = cv.imread(rr, 0).astype(np.float32)
      # img_a = cv.imread('./imgs/d.png', 0).astype(np.float32)
      img_a = cv.resize(img_a, (120, 60), interpolation = cv.INTER_AREA)
      out = GT(img_a, kernels[0])
      F_a = findF(out)

      img_b = cv.imread(path2, 0).astype(np.float32)
      # img_b = cv.imread('./imgs/e.png', 0).astype(np.float32)
      img_b = cv.resize(img_b, (120, 60), interpolation = cv.INTER_AREA)
      out = GT(img_b, kernels[0])
      F_b = findF(out)

      cal1 = 0
      for i in range(len(F_a)):
          # print(F_a[i], F_b[i], F_c[i])
          cal1 = cal1 + abs(F_a[i] - F_b[i])
      print('Each similarity: ', float(100 - cal1/10))

      ss = ss + float(100 - cal1/10)


    print("\nFinal Signature Similarity: ", float(ss/4), '\n')

    return float(ss/4)

