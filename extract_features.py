import os
import glob
import numpy as np
import cv2

def main():
    imgdir = '/Users/1plus9/Dropbox (GaTech)/Data/All gen3 data used for the paper'
    outfile = 'features.csv'

    # get all the image files
    imgfiles = glob.glob(os.path.join(imgdir, '*', '*.png'))
    imgfiles += glob.glob(os.path.join(imgdir, '*', '*.tif'))

    dirnames = []
    filenames = []
    stds = []
    means = []

    for imgf in imgfiles:
        img = cv2.imread(imgf)
        # get the pixel intensity variance
        std = np.std(img)
        # get the pixel intensity mean
        mean = np.mean(img)

        dirnames.append(os.path.basename(os.path.dirname(imgf)))
        filenames.append(os.path.basename(imgf)[:-4])
        stds.append(std)
        means.append(mean)
    
    # save the features to a csv file
    with open(outfile, 'w') as f:
        f.write('dirname,filename,std,mean\n')
        for dirname, filename, std, mean in zip(dirnames, filenames, stds, means):
            f.write('{},{},{},{}\n'.format(dirname, filename, std, mean))

if __name__ == '__main__':
    main()
