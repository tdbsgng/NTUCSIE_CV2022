import cv2
import copy
import numpy as np
def show(name,lena):
    #cv2.imshow(name, lena )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(name,lena)

def xygenerator(size,nocenter =0 ):
    for row in range(-(size//2),size//2+1):
        for col in range(-(size//2),size//2+1):
            if (row,col) != (0,0) or not nocenter:
                yield (row,col)

def zero_crossing(lena, kernel, threshold):
    dsize = lena.shape[0] - original_shape[0]
    tmp = np.zeros(original_shape,np.int8)
    output = np.zeros(original_shape,np.uint8)
    for row in range(output.shape[0]): 
        for col in range(output.shape[1]):
            value = 0
            for drow,dcol in kernel.keys(): #kernel = {(drow,dcol):weight}
                value += lena[row+drow+dsize//2][col+dcol+dsize//2]*kernel[(drow,dcol)]
            if value >= threshold:
                tmp[row][col] = 1
            elif value <= -threshold:
                tmp[row][col] = -1
            else:
                tmp[row][col] = 0
    for row in range(output.shape[0]): 
        for col in range(output.shape[1]):
            if tmp[row][col] != 1:
                output[row][col] = 255
            else :
                flag = 0 
                g = xygenerator(3,1)
                for drow,dcol in g:
                    try:
                        if tmp[row+drow][col+dcol] == -1:
                            if row+drow not in range(output.shape[0]) or row+dcol not in range(output.shape[0]):
                                print(row+drow,col+dcol)
                            flag = 1
                    except:
                        continue
                output[row][col] = 0 if flag else 255 
    return output

lena = cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
original_shape = lena.shape

def main():
    lm1_kernel = {}
    lm2_kernel = {}
    mvl_kernel = {}
    log_kernel = {}
    dog_kernel = {}
    kernel_list = [lm1_kernel, lm2_kernel, mvl_kernel, log_kernel, dog_kernel]
    lm1_weights = [
        0, 1, 0,
        1,-4, 1,
        0, 1, 0
    ]
    lm2_weights = [
        1/3, 1/3, 1/3,
        1/3,-8/3, 1/3,
        1/3, 1/3, 1/3
    ]
    mvl_weights = [
        2/3, -1/3, 2/3,
       -1/3, -4/3,-1/3,
        2/3, -1/3, 2/3    
    ]
    log_weights = [
         0,  0,   0,  -1,  -1,  -2,  -1,  -1,   0,  0,  0,
         0,  0,  -2,  -4,  -8,  -9,  -8,  -4,  -2,  0,  0,
         0, -2,  -7, -15, -22, -23, -22, -15,  -7, -2,  0,
        -1, -4, -15, -24, -14,  -1, -14, -24, -15, -4, -1,
        -1, -8, -22, -14,  52, 103,  52, -14, -22, -8, -1,
        -2, -9, -23,  -1, 103, 178, 103,  -1, -23, -9, -2,
        -1, -8, -22, -14,  52, 103,  52, -14, -22, -8, -1,
        -1, -4, -15, -24, -14,  -1, -14, -24, -15, -4, -1,
         0, -2,  -7, -15, -22, -23, -22, -15,  -7, -2,  0,
         0,  0,  -2,  -4,  -8,  -9,  -8,  -4,  -2,  0,  0,
         0,  0,   0,  -1,  -1,  -2,  -1,  -1,   0,  0,  0
    ]
    dog_weights = [
        -1,  -3,  -4,  -6,  -7,  -8,  -7,  -6,  -4,  -3, -1,
        -3,  -5,  -8, -11, -13, -13, -13, -11,  -8,  -5, -3,
        -4,  -8, -12, -16, -17, -17, -17, -16, -12,  -8, -4,
        -6, -11, -16, -16,   0,  15,   0, -16, -16, -11, -6,
        -7, -13, -17,   0,  85, 160,  85,   0, -17, -13, -7,
        -8, -13, -17,  15, 160, 283, 160,  15, -17, -13, -8,
        -7, -13, -17,   0,  85, 160,  85,   0, -17, -13, -7,
        -6, -11, -16, -16,   0,  15,   0, -16, -16, -11, -6,
        4,  -8, -12, -16, -17, -17, -17, -16, -12,  -8,  -4,
        -3,  -5,  -8, -11, -13, -13, -13, -11,  -8,  -5, -3,
        -1,  -3,  -4,  -6,  -7,  -8,  -7,  -6,  -4,  -3, -1
    ]
    weight_list = [lm1_weights, lm2_weights, mvl_weights, log_weights, dog_weights]
    for index,kernel in enumerate(kernel_list):
        weights = weight_list[index]
        g = xygenerator(int((len(weights))**0.5))
        count = 0
        for x,y in g:
            kernel[(x,y)] = weights[count]
            count += 1

    lena3x3 = cv2.copyMakeBorder(lena,1,1,1,1,cv2.BORDER_REPLICATE)
    lena11x11 = cv2.copyMakeBorder(lena,5,5,5,5,cv2.BORDER_REPLICATE)
    #show("lm1.jpg",zero_crossing(lena3x3,lm1_kernel,15))
    #show("lm2.jpg",zero_crossing(lena3x3,lm2_kernel,15))
    #show("mvl.jpg",zero_crossing(lena3x3,mvl_kernel,20))
    #show("log.jpg",zero_crossing(lena11x11,log_kernel,3000))
    show("dog.jpg",zero_crossing(lena11x11,dog_kernel,1))


if __name__ == "__main__":
    main()
