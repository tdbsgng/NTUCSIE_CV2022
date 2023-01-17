from pickle import NONE
import cv2
import copy
import numpy as np
def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite(name,lena)

def binary(lena):
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if lena[row][col] >= 128:
                lena[row][col] = 255
            else:
                lena[row][col] = 0
    return lena
def downsample(lena):
    sample=np.zeros((len(lena)//8,len(lena)//8,1))
    for row in range(len(sample)):
        for col in range(len(sample[0])):
            sample[row][col] = lena[row*8][col*8]
    return sample
def padding(lena):
    padding = np.zeros((len(lena)+2,len(lena[0])+2,1))
    for row in range(1,65):
        for col in range(1,65):
            padding[row][col] = lena[row-1][col-1]
    return padding
def yokoi(lena):
    f = open('yokoi.txt','a')
    yokoi_output = [[-1 for _ in range(66)] for _ in range(66)]
    for row in range(1,65):
        for col in range(1,65):
            if lena[row][col] == 0:
                yokoi_output[row][col]== -1
                f.write('  ')
                continue
            a1 = h(lena[row][col],lena[row][col+1],lena[row-1][col+1],lena[row-1][col])
            a2 = h(lena[row][col],lena[row-1][col],lena[row-1][col-1],lena[row][col-1])
            a3 = h(lena[row][col],lena[row][col-1],lena[row+1][col-1],lena[row+1][col])
            a4 = h(lena[row][col],lena[row+1][col],lena[row+1][col+1],lena[row][col+1])
            if a1==a2==a3==a4=='r':
                yokoi_output[row][col] = 5
            else:
                count = 0
                for i in [a1,a2,a3,a4]:
                    if i=='q':
                        count+=1
                yokoi_output[row][col] = count
            if yokoi_output[row][col] > 0:
                f.write(str(yokoi_output[row][col])+' ')
            else:
                f.write('  ')
        f.write('\n')
    f.close()
    return yokoi_output

def h(b, c, d, e):
    if b == c and (d != b or e != b):
        return 'q'
    if b == c and (d == b and e == b):
        return 'r'
    return 's'

with open('yokoi.txt','w') as f:
    f.write('')
lena = cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
lena = binary(lena)
lena = downsample(lena)
lena = padding(lena) #66*66 add a border
yokoi(lena)





