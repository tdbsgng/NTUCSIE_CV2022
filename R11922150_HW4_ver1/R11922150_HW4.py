import cv2
import copy
import numpy as np

def dilation(lena,kernel):
    lena_a = copy.deepcopy(lena)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if lena[row][col] == 255:
                for x,y in kernel:
                    try:
                        lena_a[row+x][col+y] = 255
                    except:
                        continue
    return lena_a


def erosion(lena,kernel):
    lena_b = copy.deepcopy(lena)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            flag = 0
            for x,y in kernel:
                try:
                    if lena[row+x][col+y] != 255:
                        flag = 1
                        break
                except:
                    continue
            lena_b[row][col] = 0 if flag else 255
    return lena_b
def opening(lena,kernel):
    return dilation(erosion(lena,kernel),kernel)
def closing(lena,kernel):
    return erosion(dilation(lena,kernel),kernel)
def ham(lena,j,k):
    tmp = -lena + 255
    left = erosion(lena,j)
    right = erosion(tmp,k)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if left[row][col]==255 and right[row][col]==255:
                tmp[row][col] = 255 
            else:
                tmp[row][col] = 0
    return tmp


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
lena = binary(cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED))
kernel = [
          [-2, -1], [-2, 0], [-2, 1],
[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
[0, -2],  [0, -1],  [0, 0],  [0, 1],  [0, 2],
[1, -2],  [1, -1],  [1, 0],  [1, 1],  [1, 2],
          [2, -1],  [2, 0],  [2, 1]]
j = [[0, -1], [0, 0], 
              [1, 0]
]
k = [[-1, 0], [-1, 1],
              [0, 1]
]
ans_list=[]
ans_list.append((dilation(lena,kernel),'dilation.jpg'))
ans_list.append((erosion(lena,kernel),'erosion.jpg'))
ans_list.append((opening(lena,kernel),'opening.jpg'))
ans_list.append((closing(lena,kernel),'closing.jpg'))
ans_list.append((ham(lena,j,k),'ham.jpg'))
for ans in ans_list:
    show(ans[1],ans[0])






