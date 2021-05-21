
import multiresolutionimageinterface as mir
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
from shutil import copyfile
import time
import csv
import gc
import random


import concurrent.futures
tic = time.perf_counter()

inputDir = 'D:\\tumor'


def process_image(fold):
    print(fold)
    target_img = fold[:-4]
    tmg = 'D:\\patches\\' + target_img
    count = 0

    if not os.path.isdir(tmg):
        os.mkdir(tmg)


    patch_folder = 'D:\\patches\\' + target_img
    try:
        inpDirTissueMask = os.path.join('D:\\tissue_masks\\' + target_img + '_tissue_mask.tif')
        inpDirAnnoMask = os.path.join('D:\\tumor_masks\\' + target_img + '_mask.tif')
        
        reader = mir.MultiResolutionImageReader()
        imageO = reader.open(inputDir + '\\' + fold)# Original Image = imageO Saved in a different drive
        imageA = reader.open(inpDirAnnoMask) # Annotated Mask = imageA
        imageT = reader.open(inpDirTissueMask) # Tissue Mask = imageT

        if imageO is None:
            print(inputDir + '\\' + fold)    
        if imageA is None:
            print(inpDirAnnoMask)
        if imageT is None:
            print(inpDirTissueMask)
        
        
        
        target_csv = target_img + '.csv'  
        csv_path_none = patch_folder  + '\\negative.csv'
        csv_path_pos = patch_folder  + '\\positive.csv'

        with open(csv_path_none, 'a', newline='') as pl: # create csv file to save image data pixel location of the patch and its corresponding label
            nonewriter = csv.writer(pl)
            nonewriter.writerow(['WSI_name', 'WSI_level_row_col', 'Label'])
            with open(csv_path_pos, 'a', newline='') as pos:
                poswriter = csv.writer(pos)
                poswriter.writerow(['WSI_name', 'WSI_level_row_col', 'Label'])
                
                for i in range(7, -1, -1):
                    pcount = 0
                    temp = patch_folder + "\\" + str(i)
                    if not os.path.isdir(temp):
                        os.mkdir(temp)
                    
                    tip = temp + '\\PositivePatches'
                    tin = temp + '\\NegativePatches'
                    if not os.path.isdir(tip):
                        os.mkdir(tip)
                    if not os.path.isdir(tin):
                        os.mkdir(tin)


                    dimsO = imageO.getLevelDimensions(i)

                    dimsA = imageA.getLevelDimensions(i)

                    dimsT = imageT.getLevelDimensions(i)
                    ds = imageO.getLevelDownsample(i)
                    
                    for j in range(0, dimsO[0] - 512, 128):
                        for k in range(0, dimsO[1] - 512, 128):

                            temp_patchO = imageO.getUCharPatch(int(j * ds), int(k * ds), 512, 512, i)
          
                            temp_patchA = imageA.getUCharPatch(int(j * ds), int(k * ds), 512, 512, i)

                            temp_patchT = imageT.getUCharPatch(int(j * ds), int(k * ds), 512, 512, i)
                                  

                            if np.sum(temp_patchA) != 0:
                                pcount += 1
                                label = 1
                                # positive += 1
                                poswriter.writerow([target_img, 'patch_' + str(i) + '_' + str(j) + '_' + str(k), label])
                                trueposPath = patch_folder + '\\PositivePatches'
                                cv2.imwrite(tip + '\\' + 'patch_' +  str(i) + '_' + str(j) + '_' + str(k) + '_' + str(label) + '.png', temp_patchO)

                    ncount = 0
                    x = list(range(0, dimsO[0] - 512, 128))
                    y = list(range(0, dimsO[1] - 512, 128))
                    coordinate = []
                    while True:
                        
                        j = random.choice(x)
                        k = random.choice(y)
                        if (j,k) in coordinate:
                            continue
                        temp_patchO = imageO.getUCharPatch(int(j * ds), int(k * ds), 512, 512, i)

                        temp_patchA = imageA.getUCharPatch(int(j * ds), int(k * ds), 512, 512, i)

                        temp_patchT = imageT.getUCharPatch(int(j * ds), int(k * ds), 512, 512, i)

            

                        if np.sum(temp_patchT) != 0 and np.sum(temp_patchA) == 0:
                            coordinate.append((j,k))
                        
                            label = 0
                            ncount += 1
                            nonewriter.writerow([target_img, 'patch_' + str(i) + '_' + str(j) + '_' + str(k), label])
                            negPath = patch_folder + '\\NegativePatches'
                            cv2.imwrite(tin + '\\' + 'patch_'  + str(i) + '_' + str(j) + '_' + str(k) + '_' + str(label) + '.png', temp_patchO)
                        
                            
                            
                        if pcount == ncount:
                            break
                            
                    print(f'Resolution level = {i}\nPositive count = {pcount}\nNegative count = {ncount}\n')

    except Exception as e:
        print('Exception is', e)

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_image,os.listdir(inputDir))

    toc = time.perf_counter()
    TotalTimeReq = toc-tic
    print("Total Time Taken: ",TotalTimeReq)




