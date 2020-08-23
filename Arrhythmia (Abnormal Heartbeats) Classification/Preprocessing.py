import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import wfdb

def Get_Path():
    paths = glob('/Data/*.atr')
    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()
    return paths


def segmentation(records,beat):
    Normal = []
    for e in records:
        signals, fields = wfdb.rdsamp(e)
        ann = wfdb.rdann(e, 'atr')
        good = [beat]
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                Normal.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
    return Normal


 paths = Get_Path()
Annotations = ['A', 'E', 'L', 'N', '/', 'R', 'V']

# Following are the annotations for beat categories
# A    = Atrial premature contraction    
# V    = Ventricular premature contraction
# /    = Paced beat
# E    = Ventricular escape beat
# L    = Left bundle branch block
# R    = Right bundle branch block
# N    = Normal beats

for i in Annotations:
	Segmented = segmentation(records, i)
		for count, i in enumerate(Segmented):
			fig = plt.figure(frameon=False)
			plt.plot(i) 
			plt.xticks([]), plt.yticks([])
			for spine in plt.gca().spines.values():
				spine.set_visible(False)

			filename = '/content/data' + i + '/' + str(count)+'.png'
			fig.savefig(filename)
			im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
			im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
			cv2.imwrite(filename, im_gray)


#Data Augmentation
def cropping(image, filename):
    
    #Left Top Crop
    crop = image[:96, :96]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'leftTop' + '.png', crop)
    
    #Center Top Crop
    crop = image[:96, 16:112]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'centerTop' + '.png', crop)
    
    #Right Top Crop
    crop = image[:96, 32:]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'rightTop' + '.png', crop)
    
    #Left Center Crop
    crop = image[16:112, :96]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'leftCenter' + '.png', crop)
    
    #Center Center Crop
    crop = image[16:112, 16:112]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'centerCenter' + '.png', crop)
    
    #Right Center Crop
    crop = image[16:112, 32:]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'rightCenter' + '.png', crop)
    
    #Left Bottom Crop
    crop = image[32:, :96]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'leftBottom' + '.png', crop)
    
    #Center Bottom Crop
    crop = image[32:, 16:112]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'centerBottom' + '.png', crop)
    
    #Right Bottom Crop
    crop = image[32:, 32:]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(filename[:-4] + 'rightBottom' + '.png', crop)


#Run this code section for all the classes

# A    = Atrial premature contraction    
# V    = Ventricular premature contraction
# P    = Paced beat
# E    = Ventricular escape beat
# L    = Left bundle branch block
# R    = Right bundle branch block
# N    = Normal beats


paths = glob('/content/Data/Train/A/*.png')
paths.sort()
for path in paths:
	image = cv2.imread(path)
    cropping(image,path)