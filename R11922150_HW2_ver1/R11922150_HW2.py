from msilib.schema import Component
import cv2
import copy
import numpy as np
import matplotlib.pyplot  as plt
def a(lena):
    lena_a=copy.deepcopy(lena)
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if lena_a[row][col] >= 128:
                lena_a[row][col] = 255
            else:
                lena_a[row][col] = 0
    cv2.imwrite('a.jpg', lena_a)
    show('a.jpg', lena_a)
    return lena_a

def b(lena):
    h=[0 for i in range(256)]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            h[lena[row][col]]+=1
    intensity = [f'{i}' for i in range(256)]
    x = np.arange(256)
    plt.bar(x, h, color=['red' for _ in range(256)],width=1.0)
    plt.xlabel('Grayscale intensity')
    plt.ylabel('Frequency')
    plt.title('Historgram')
    plt.savefig('b.png')
    plt.show()

def c(lena):
    flag,count,direction=1,1,1
    matrix=[[float("inf") for _ in range(len(lena[0]))] for _ in range(len(lena))]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if lena[row][col]:
                matrix[row][col]=count
                count+=1
    print("processing Iterative Algorithm")
    while flag:
        flag=0
        if direction:
            for row in range(len(lena)):
                for col in range(len(lena[0])):
                    if lena[row][col]:
                        tmp=matrix[row][col]
                        matrix[row][col]=min([matrix[row][col]]+connect4(lena,matrix,row,col,direction))
                        if tmp!=matrix[row][col]:
                            flag=1
        else:
            for row in range(len(lena)-1,-1,-1):
                for col in range(len(lena[0])-1,-1,-1):
                    if lena[row][col]:
                        tmp=matrix[row][col]
                        matrix[row][col]=min([matrix[row][col]]+connect4(lena,matrix,row,col,direction))
                        if tmp!=matrix[row][col]:
                            flag=1

        direction=1-direction
    component=[0 for _ in range(count)]
    target=[]          #component whose area more than 500
    for row in matrix:
        for index in row:
            if type(index)==int:
                component[index]+=1
                if component[index]==500:
                    target.append(index)
    print(f'component index whose area greater than 500  :  {target}')
    target_list=[[] for _ in target]
    answer_list=[[] for _ in target]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            if matrix[row][col] in target:
                target_list[target.index(matrix[row][col])].append((row,col))
    for index,list in enumerate(target_list):
        min_row,min_col=float('inf'),float('inf')
        max_row,max_col=-1,-1
        total=[0,0]
        for row,col in list:
            min_row,min_col,max_row,max_col=min(min_row,row),min(min_col,col),max(max_row,row),max(max_col,col)
            total[0]+=col
            total[1]+=row
        center=total[0]//len(list),total[1]//len(list)
        answer_list[index]+=[(min_col,min_row),(max_col,max_row),center]
    print('format : (left-top),(right-bottom),center)\n',answer_list)
    lena = cv2.cvtColor(lena, cv2.COLOR_GRAY2BGR)
    for answer in answer_list:
        cv2.rectangle(lena,answer[0],answer[1],(0,0,255),3)
        cv2.line(lena, (answer[2][0]-10,answer[2][1]), (answer[2][0]+10,answer[2][1]), (0,0,255), 2)
        cv2.line(lena, (answer[2][0],answer[2][1]-10), (answer[2][0],answer[2][1]+10), (0,0,255), 2)
        show('bounding box',lena)
    cv2.imwrite('c.jpg', lena)



def connect4(image,matrix,row,col,direction):
    l=[]
    if direction: #top down
        try:
            if image[row-1][col]:
                l.append(matrix[row-1][col])
        except:
            pass
        try:
            if image[row][col-1]:
                l.append(matrix[row][col-1])
        except:
            pass
    else:
        try:
            if image[row+1][col]:
                l.append(matrix[row+1][col])
        except:
            pass
        try:
            if image[row][col+1]:
                l.append(matrix[row][col+1])
        except:
            pass
    return l
def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

lena=cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
lena_a=a(lena)
b(lena)
c(lena_a)






