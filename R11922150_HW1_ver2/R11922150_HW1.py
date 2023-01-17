import cv2
import copy
import numpy as np
from scipy.ndimage import rotate
def part1_a(lena):
    lena_a=copy.deepcopy(lena)
    for index in range(len(lena_a)):
        lena_a[index]=lena[len(lena)-1-index]
    return lena_a

def part1_b(lena):
    lena_b=copy.deepcopy(lena)
    for row_index in range(len(lena)):
        for col_index in range(len(lena[0])):
            lena_b[row_index][col_index]=lena[row_index][len(lena[0])-1-col_index]
    return lena_b    

def part1_c(lena):
    lena_c=copy.deepcopy(lena)
    for row_index in range(len(lena)):
            for col_index in range(len(lena[0])):
                lena_c[row_index][col_index]=lena[col_index][row_index]
    return lena_c

def part2_d(lena):
    lena_d=rotate(lena,angle=-45)
    return lena_d

def part2_e(lena):
    lena_e=cv2.resize(lena,(256,256))
    return lena_e
def part2_f(lena):
    lena_f=copy.deepcopy(lena)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if lena_f[row][col] > 128:
                lena_f[row][col] = 255
            else:
                lena_f[row][col] = 0
    return lena_f
def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

lena=cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
cv2.imwrite('part1_a.jpg', part1_a(lena))
cv2.imwrite('part1_b.jpg', part1_b(lena))
cv2.imwrite('part1_c.jpg', part1_c(lena))
cv2.imwrite('part2_d.jpg', part2_d(lena))
cv2.imwrite('part2_e.jpg', part2_e(lena))
cv2.imwrite('part2_f.jpg', part2_f(lena))
show('part1_a.jpg', part1_a(lena))
show('part1_b.jpg', part1_b(lena))
show('part1_c.jpg', part1_c(lena))
show('part2_d.jpg', part2_d(lena))
show('part2_e.jpg', part2_e(lena))
show('part2_f.jpg', part2_f(lena))

