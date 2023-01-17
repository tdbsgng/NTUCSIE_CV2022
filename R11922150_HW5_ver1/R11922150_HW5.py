import cv2
import copy
import numpy as np

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

def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name,lena)

lena = cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
kernel = [
          [-2, -1], [-2, 0], [-2, 1],
[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
[0, -2],  [0, -1],  [0, 0],  [0, 1],  [0, 2],
[1, -2],  [1, -1],  [1, 0],  [1, 1],  [1, 2],
          [2, -1],  [2, 0],  [2, 1]]

ans_list=[]
ans_list.append((dilation(lena,kernel),'dilation.jpg'))
ans_list.append((erosion(lena,kernel),'erosion.jpg'))
ans_list.append((opening(lena,kernel),'opening.jpg'))
ans_list.append((closing(lena,kernel),'closing.jpg'))
for ans in ans_list:
    show(ans[1],ans[0])






