import cv2
import copy
import numpy as np
def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name,lena)
def xygenerator(size):
    for row in range(-(size//2),size//2+1):
        for col in range(-(size//2),size//2+1):
            yield (row,col)
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

def edge_detection(lena, filters, threshold):
    dsize = lena.shape[0] - original_shape[0]
    output = np.zeros(original_shape,np.uint8)
    for row in range(output.shape[0]): 
        for col in range(output.shape[1]):
            value_list = []
            for filter in filters:
                value = 0
                for drow,dcol in filter.keys(): #filter = {(drow,dcol):weight}
                    value += lena[row+drow+dsize//2][col+dcol+dsize//2]*filter[(drow,dcol)]
                value_list.append(value)
            if len(value_list) == 2 :
                output[row][col] = 0 if (value_list[0]**2+value_list[1]**2)**0.5 >= threshold else 255
            else:
                value = max(value_list)
                output[row][col] = 0 if value >= threshold else 255
    return output


lena = cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
original_shape = lena.shape

def main():
    roberts_filters = [ 
        { (0,0):-1, (1,1):1 }, #check
        { (0,1):-1, (1,0):1 }
    ]
    prewitt_filters = [
        { (-1,-1):-1, (-1,0):-1, (-1,1):-1, (1,-1):1, (1,0):1, (1,1):1 },
        { (-1,-1):-1, (0,-1):-1, (1,-1):-1, (-1,1):1, (0,1):1, (1,1):1 }
    ]
    sobel_filters = [
        { (-1,-1):-1, (-1,0):-2, (-1,1):-1, (1,-1):1, (1,0):2, (1,1):1 },
        { (-1,-1):-1, (0,-1):-2, (1,-1):-1, (-1,1):1, (0,1):2, (1,1):1 }
    ]
    fnc_filters = [
        { (-1,-1):-1, (-1,0):-(2**0.5), (-1,1):-1, (1,-1):1, (1,0):2**0.5, (1,1):1 },
        { (-1,-1):-1, (0,-1):-(2**0.5), (1,-1):-1, (-1,1):1, (0,1):2**0.5, (1,1):1 }
    ]
    kirsch_filters = [
        { (-1,-1):-3, (-1,0):-3, (-1,1):5, (0,-1):-3, (0,1):5, (1,-1):-3, (1,0):-3, (1,1):5 },

        { (-1,-1):-3, (-1,0):5, (-1,1):5, (0,-1):-3, (0,1):5, (1,-1):-3, (1,0):-3, (1,1):-3 },

        { (-1,-1):5, (-1,0):5, (-1,1):5, (0,-1):-3, (0,1):-3, (1,-1):-3, (1,0):-3, (1,1):-3 },

        { (-1,-1):5, (-1,0):5, (-1,1):-3, (0,-1):5, (0,1):-3, (1,-1):-3, (1,0):-3, (1,1):-3 },

        { (-1,-1):5, (-1,0):-3, (-1,1):-3, (0,-1):5, (0,1):-3, (1,-1):5, (1,0):-3, (1,1):-3 },

        { (-1,-1):-3, (-1,0):-3, (-1,1):-3, (0,-1):5, (0,1):-3, (1,-1):5, (1,0):5, (1,1):-3 },

        { (-1,-1):-3, (-1,0):-3, (-1,1):-3, (0,-1):-3, (0,1):-3, (1,-1):5, (1,0):5, (1,1):5 },

        { (-1,-1):-3, (-1,0):-3, (-1,1):-3, (0,-1):-3, (0,1):5, (1,-1):-3, (1,0):5, (1,1):5 }
    ]
    robinson_filters = [ #check
        { (-1,-1):-1, (-1,0):0, (-1,1):1, (0,-1):-2, (0,1):2, (1,-1):-1, (1,0):0, (1,1):1 },

        { (-1,-1):0, (-1,0):1, (-1,1):2, (0,-1):-1, (0,1):1, (1,-1):-2, (1,0):-1, (1,1):0 },

        { (-1,-1):1, (-1,0):2, (-1,1):1, (0,-1):0, (0,1):0, (1,-1):-1, (1,0):-2, (1,1):-1 },

        { (-1,-1):2, (-1,0):1, (-1,1):0, (0,-1):1, (0,1):-1, (1,-1):0, (1,0):-1, (1,1):-2 }
    ]
    robinson_filters.extend([{(x,y):-val for (x,y),val in filter.items()} for filter in robinson_filters])
    nb_filters = []
    nb_weights = [
        [100, 100, 100, 100, 100,
         100, 100, 100, 100, 100,
          0,   0 ,  0 ,  0 ,  0 ,
        -100,-100,-100,-100,-100,
        -100,-100,-100,-100,-100],

        [100, 100, 100, 100, 100,
         100, 100, 100, 78 , -32,
         100,  92 , 0 ,-92 ,-100,
          32, -78,-100,-100,-100,
        -100,-100,-100,-100,-100],

        [100, 100, 100, 32, -100,
         100, 100, 92 , -78,-100,
         100, 100 , 0 ,-100,-100,
         100, 78, -92 ,-100,-100,
         100, -32,-100,-100,-100],

        [-100, -100, 0, 100, 100,
         -100, -100, 0, 100, 100,
         -100, -100, 0, 100 ,100,
         -100, -100, 0, 100, 100,
         -100, -100, 0, 100, 100],

        [-100,  32, 100, 100, 100,
         -100, -78, 92, 100, 100,
         -100, -100, 0, 100 ,100,
         -100, -100, -92, 78, 100,
         -100, -100, -100, -32, 100],

         [100, 100, 100, 100, 100,
         -32, 78, 100, 100, 100,
         -100, -92 , 0 ,92 ,100,
        -100,-100,-100,-78,32,
        -100,-100,-100,-100,-100]

    ]
    for i in range(6):
        g = xygenerator(5)
        dic = {}
        count = 0
        for x,y in g:
            dic[(x,y)] = nb_weights[i][count]
            count += 1
        nb_filters.append(dic)
    lena_pad1 = padding(lena)
    lena_pad2 = padding(lena_pad1)
    show("roberts.jpg", edge_detection(lena_pad1, roberts_filters, 30))
    show("prewitt.jpg", edge_detection(lena_pad1, prewitt_filters, 24))
    show("sobel.jpg", edge_detection(lena_pad1, sobel_filters, 38))
    show("fnc.jpg", edge_detection(lena_pad1, fnc_filters, 30))
    show("kirsch.jpg", edge_detection(lena_pad1, kirsch_filters, 135))
    show("robinson.jpg", edge_detection(lena_pad1, robinson_filters, 43))
    show("nb.jpg", edge_detection(lena_pad2, nb_filters, 12500))

if __name__ == "__main__":
    main()





