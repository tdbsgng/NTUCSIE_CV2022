from pickle import NONE
import cv2
import copy
import numpy as np
def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name,lena)

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

def pair_relationship(yokoi):
    pr = [[-1 for _ in range(66)] for _ in range(66)]
    for row in range(1,65):
        for col in range(1,65):
            if yokoi[row][col] >= 1:
                if yokoi[row][col] !=1:
                    pr[row][col] = 'q'
                    continue
                sum = 0
                for x,y in [(-1,0),(1,0),(0,1),(0,-1)]:
                    h_value = 1 if yokoi[row+x][col+y] == 1 else 0
                    sum += h_value
                pr[row][col] = 'p' if sum>=1 else 'q'
    return pr
def h1(b, c, d, e):
    if b==c and (d!=b or e!=b):
        return 1
    else:
        return 0 

def thinning(lena):
    last_output = None
    current_output = lena
    while True:
        last_output = copy.deepcopy(current_output)
        yokoi_output = yokoi(current_output)
        pr_output = pair_relationship(yokoi_output)
        for row in range(1,65):
            for col in range(1,65):
                if pr_output[row][col] != 'p':
                    current_output[row][col] = last_output[row][col]
                else:
                    a1 = h1(current_output[row][col],current_output[row][col+1],current_output[row-1][col+1],current_output[row-1][col])
                    a2 = h1(current_output[row][col],current_output[row-1][col],current_output[row-1][col-1],current_output[row][col-1])
                    a3 = h1(current_output[row][col],current_output[row][col-1],current_output[row+1][col-1],current_output[row+1][col])
                    a4 = h1(current_output[row][col],current_output[row+1][col],current_output[row+1][col+1],current_output[row][col+1])    
                    if a1+a2+a3+a4 ==1:
                        current_output[row][col] = 0 
                    else:
                        current_output[row][col] = last_output[row][col]
        if (current_output == last_output).all():
            return current_output
with open('yokoi.txt','w') as f:
    f.write('')
lena = cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
lena = binary(lena)
lena = downsample(lena)
lena = padding(lena) #66*66 add a border
show('thinning.jpg',thinning(lena))






