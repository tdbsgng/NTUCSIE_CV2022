from msilib.schema import Component
import cv2
import copy
import numpy as np
import matplotlib.pyplot  as plt

def a(lena):
    h=[0 for i in range(256)]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            h[lena[row][col]]+=1
    show('a.jpg',lena)
    #histogram
    intensity = [f'{i}' for i in range(256)]
    x = np.arange(256)
    plt.bar(x, h, color=['red' for _ in range(256)],width=1.0)
    plt.xlabel('Grayscale intensity')
    plt.ylabel('Frequency')
    plt.title('(a) Historgram')
    plt.savefig('a_histogram.jpg')
    plt.show()

def b(lena):
    h=[0 for i in range(256)]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            lena[row][col]//=3
            h[lena[row][col]]+=1
    show('b.jpg',lena)
    #histogram
    intensity = [f'{i}' for i in range(256)]
    x = np.arange(256)
    plt.bar(x, h, color=['red' for _ in range(256)],width=1.0)
    plt.xlabel('Grayscale intensity')
    plt.ylabel('Frequency')
    plt.title('(b) Historgram')
    plt.savefig('b_histogram.jpg')
    plt.show()
def c(lena):
    h=[0 for i in range(256)]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            h[lena[row][col]]+=1
    c=[sum(h[:i+1]) for i in range(256)]
    sk=[round((c[i]*255)/(512*512)) for i in range(256)]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            lena[row][col]=sk[lena[row][col]]
    show('c.jpg',lena)
    h=[0 for i in range(256)]
    for row in range(len(lena)):
        for col in range(len(lena[0])):
            h[lena[row][col]]+=1
    #histogram
    intensity = [f'{i}' for i in range(256)]
    x = np.arange(256)
    plt.bar(x, h, color=['red' for _ in range(256)],width=1.0)
    plt.xlabel('Grayscale intensity')
    plt.ylabel('Frequency')
    plt.title('(c) Historgram')
    plt.savefig('c_histogram.jpg')
    plt.show()


def show(name,lena):
    cv2.imshow(name, lena )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name,lena)


lena=cv2.imread("lena.bmp",cv2.IMREAD_UNCHANGED)
a(lena)
b(lena)
c(lena)






