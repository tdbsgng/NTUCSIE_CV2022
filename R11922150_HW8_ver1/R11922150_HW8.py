import cv2
import copy
import numpy as np
import random
from math import log10 as log
def show(name,lena):
    #cv2.imshow(name, lena )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(name,lena)

def gaussian(lena, amplitude):
    gaussian_lena = copy.deepcopy(lena)
    for row in range(gaussian_lena.shape[0]):
        for col in range(gaussian_lena.shape[1]):
            gaussian_lena[row][col] = min(255, int(lena[row][col] + amplitude*random.gauss(0,1)))
    return gaussian_lena

def snp(lena, threshold):
    snp_lena =  copy.deepcopy(lena)
    for row in range(snp_lena.shape[0]):
        for col in range(snp_lena.shape[1]):
            random_value = random.uniform(0,1)
            if random_value <= threshold:
                snp_lena[row][col] = 0
            elif random_value >= 1-threshold:
                snp_lena[row][col] = 255
            else:
                snp_lena[row][col] = lena[row][col]
    return snp_lena

def padding(lena):
    pad_lena = np.zeros((lena.shape[0]+2 ,lena.shape[1]+2),np.uint8)
    pad_lena[0][0] = lena[0][0]
    pad_lena[-1][0] = lena[-1][0]
    pad_lena[0][-1] = lena[0][-1]
    pad_lena[-1][-1] = lena[-1][-1]
    for row in range(lena.shape[0]):
        for col in range(lena.shape[1]):
            pad_lena[row+1][col+1] = lena[row][col]
    for row in range(1,pad_lena.shape[0]-1):
        pad_lena[row][0] = lena[row-1][0]
        pad_lena[row][-1] = lena[row-1][-1]
    for col in range(1,pad_lena.shape[1]-1):
        pad_lena[0][col] = lena[0][col-1]
        pad_lena[-1][col] = lena[-1][col-1]
    return pad_lena

def box(lena, kernel):
    size = int((len(kernel))**0.5)
    output = np.zeros((lena.shape[0]-size+1 ,lena.shape[1]-size+1),np.uint8)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            value = 0 
            for x,y in kernel:
                value += lena[row+size//2+x][col+size//2+y]
            output[row][col] = value//len(kernel)
    return output

def median(lena, kernel):
    size = int((len(kernel))**0.5)
    output = np.zeros((lena.shape[0]-size+1 ,lena.shape[1]-size+1),np.uint8)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            tmp = []
            for x,y in kernel:
                tmp.append(lena[row+size//2+x][col+size//2+y])
            tmp.sort()
            output[row][col] = tmp[len(kernel)//2]
    return output

def dilation(lena,kernel):
    lena_a = copy.deepcopy(lena)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            maximum = -1
            for x,y in kernel:
                try:
                    if lena[row+x][col+y] > maximum:
                        maximum = lena[row+x][col+y]
                except:
                    continue
            lena_a[row][col] = maximum
    return lena_a


def erosion(lena,kernel):
    lena_b = copy.deepcopy(lena)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            minimum = 256
            for x,y in kernel:
                try:
                    if lena[row+x][col+y] < minimum:
                        minimum = lena[row+x][col+y]
                except:
                    continue
            lena_b[row][col] = minimum
    return lena_b
def opening(lena,kernel):
    return dilation(erosion(lena,kernel),kernel)
def closing(lena,kernel):
    return erosion(dilation(lena,kernel),kernel)
def snr(lena,noise_lena):
    lena = lena / 255
    noise_lena = noise_lena / 255
    return round(20*log(np.std(lena)/np.std((noise_lena-lena))),3)




def main(): 
    random.seed(123)
    noise_images = []
    lena = cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
    noise_images.append(["gaussian10",gaussian(lena,10)])
    noise_images.append(["gaussian30",gaussian(lena,30)])
    noise_images.append(["snp005",snp(lena,0.05)])
    noise_images.append(["snp01",snp(lena,0.1)])
    #test = cv2.imread("median_5x5.bmp",cv2.IMREAD_UNCHANGED)
    #print(snr(lena,test))
    #return
    kernel_disk = [
            [-2, -1], [-2, 0], [-2, 1],
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [0, -2],  [0, -1],  [0, 0],  [0, 1],  [0, 2],
    [1, -2],  [1, -1],  [1, 0],  [1, 1],  [1, 2],
            [2, -1],  [2, 0],  [2, 1]
    ]
    kernel_3x3 = [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ]
    kernel_5x5 = [
    [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [0, -2],  [0, -1],  [0, 0],  [0, 1],  [0, 2],
    [1, -2],  [1, -1],  [1, 0],  [1, 1],  [1, 2],
    [2, -2],  [2, -1],  [2, 0],  [2, 1],  [2, 2]
    ]
    for name, noise_image in noise_images:
        show(f"{name}_{str(snr(lena,noise_image))}.jpg",noise_image)
        noise_514 = padding(noise_image) #for 3x3
        print(f'processing {name} 3x3box')
        box3x3 = box(noise_514,kernel_3x3)
        show(f"{name}_3x3box_{str(snr(lena,box3x3))}.jpg",box3x3)
        print(f'processing {name} 3x3median')
        median3x3 = median(noise_514,kernel_3x3)
        show(f"{name}_3x3median_{str(snr(lena,median3x3))}.jpg",median3x3)
        noise_516 = padding(noise_514) #for 5x5
        print(f'processing {name} 5x5box') 
        box5x5 = box(noise_516,kernel_5x5)       
        show(f"{name}_5x5box_{str(snr(lena,box5x5))}.jpg",box5x5)
        print(f'processing {name} 5x5median')
        median5x5 = median(noise_516,kernel_5x5)
        show(f"{name}_5x5median_{str(snr(lena,median5x5))}.jpg",median5x5)
        print(f'processing {name} open then close') 
        otc = closing(opening(noise_image,kernel_disk),kernel_disk)
        show(f"{name}_otc_{str(snr(lena,otc))}.jpg",otc)
        print(f'processing {name} close then open')
        cto = opening(closing(noise_image,kernel_disk),kernel_disk)
        show(f"{name}_cto_{str(snr(lena,cto))}.jpg",cto)

if __name__ =="__main__":
    main()





