#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from PIL import Image
import pprint
import math

#이미지 파일 불러옴 
cur_path = os.getcwd()

print(cur_path)

img_name1 = "0121.jpg"
image_path1 = os.path.join(cur_path, img_name1)
img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#print(img1)
#cv2.imshow('image', img1)

print('this is img1 : ' , type(img1))

img_list = []
img_list = [[255 for col in range(32)] for row in range(32)]

#직선 그리기 
#대각선
# 삼각형 그리기
'''
for i in range(len(listasd)):
    listasd[i][i] = 0
  
for i in range(len(list123)):
    list123[i][len(list)-i-1] = 0
'''
for i in range(len(img_list)):
    img_list[i][28] = 0

'''
for i in range(len(list123)):
    list123[28][i] = 0
'''            
#list를 이미지로 바꾸기
array = np.array(img1)

image = Image.fromarray(array)
image.show()

# r,theta로 변환하기 위한 테이블 만들기 
# 원본 이미지에서 0인 것의 좌표만 가져옴 
# 검은색 = 밝기가 0 이므로

#검은색 픽셀의 개수를 먼저 센 후, list pos[][]를 초기화 함.
count = 0
for i in range(len(array)):
    for j in range(len(array[0])):
        if array[i][j] == 0:
            count += 1

#pos[][]는 값이 0인 픽셀의 좌표를 저장하는 list 
pos = [[0]*2 for i in range(count)]
k = 0
for i in range(len(array)):
    for j in range(len(array[0])):
        if array[i][j] == 0:
            pos[k][0] = i
            pos[k][1] = -j #수학적 xy 좌표가 아닌 이미지임을 염두
            k += 1

print("\npos : ", pos)

#theta 설정

#theta = [-45, 0, 45, 90]
#theta = np.arange(-45, 90, 1)
theta = list(range(-45,91))
r_theta = [[0]*len(theta) for i in range(len(pos))]

for i in range(len(r_theta)):
    for j in range(len(r_theta[0])):
        #r = x*cos(theta) + y*sin(theta)
        r_theta[i][j] = round(pos[i][0]*math.cos(math.pi * theta[j]/180) + pos[i][1]*math.sin(math.pi * theta[j]/180), 1)

#각 점의 좌표와 각도에 따른 r값을 저장하고 있는 배열 r_theta 출력
print("\nr_theta : ")
for i in range(len(r_theta)):
    print( r_theta[i][0], r_theta[i][1], r_theta[i][2], r_theta[i][3])

#r값 정렬하기
temp = []
for i in range(len(r_theta)):
    for j in range(len(r_theta[0])):
        temp.append(r_theta[i][j])
        
#r 값을 저장한 배열 temp에서 중복을 제거하여 r_overlap_eli 라는 배열에 입력         
temp = set(temp)
temp = sorted(temp)   

print("\n정렬 이후의 r 값 출력 : ")
for i in range(len(temp)):
    print(temp[i])

#누적 테이블 만들기 
#r_theta와 temp를 이용
acc = [[0]*len(temp) for i in range(len(theta))]

count = 0
for i in range(len(acc)): #len(acc) = len(r_theta[0])
    for j in range(len(acc[0])): # len(acc[0] = len(temp))
        count = 0
        for k in range(len(r_theta)):
            if r_theta[k][i] == temp[j]: #temp는 r값이 정렬되어있는 배열
                count += 1
        acc[i][j] = count     
        
print("\n누적 테이블 : ")
for i in range(len(acc)):
    for j in range(len(acc[0])):
        print(acc[i][j], end = ' ')
    print("\n")
        

#누적 테이블에서 thersholld를 통과한 값의 r과 theta 찾기
#threshold 값을 설정하고 threshold를 넘는 것이 몇개가 있는지 세기 
threshold = 2
count = 0

for i in range(len(acc)):
    for j in range(len(acc[0])):
        if acc[i][j] > threshold:
            count += 1

max = [[0]*3 for i in range(count)] # maximum value, max = [누적 값, 세타 값, r 값]

k = 0
for i in range(len(acc)):
    for j in range(len(acc[0])):
        if acc[i][j] > threshold:
            max[k] = [acc[i][j], theta[i], temp[j]]
            k += 1

print("\n max : ", max)

#find r, theta 
#xcos(theta) + ysin(theta) = r
#the equation is y = x

for i in range(len(max)):
    print("\n r, theta is : ", max[i][2], max[i][1])
        
#now what i have to do is draw the line again using r, theta
#좌표평면에 나타내기

axes = plt.gca()
axes.set_ylim([-32,0])
axes.set_xlim([0,32])

x = np.arange(0, 32, 0.01)
def line(x, r, theta):
    return (r - x * np.cos(np.pi * theta/180)) / np.sin(np.pi * theta/180) 

for i in range(len(max)):  
    if max[i][1] == 90: # theta가 90이면 실제 이미지 상에서는 theta = 0임
        #plt.axvline(x = -max[i][2])
        plt.plot(x,line(x,max[i][2],90))
    elif max[i][1] == 0: # 반대로 theta가 0이면 실제 이미지 상에서는 theta = 90임
        #plt.plot(x, line(x, -max[i][2], 90))
        plt.axvline(x = max[i][2])
    else:
        plt.plot(x, line(x, max[i][2], max[i][1]))

   
#math.cos(math.pi * theta[j]/180)
#plt.plot(x, y)

#print("\nmax[1] : ", max[1], "max[2] : " , max[2])

#r, theta 그래프 여러개를 하나의 좌표평면 위에 나타내기 
#def f(t1, t2):
#    theta = np.arange(-90, 90, 0.01) #theta 범위
#    return t1 * math.cos(theta) + t2 * math.sin(theta)

#for i in range(len(pos)):
#    t.append(f(pos[i,0], pos[i, 1]))

#theta = range(-90, 90)
#r = [math.cos(math.pi * t / 180) + math.sin(math.pi * t / 180 for t in range(-90,90))]
'''
def f(x, y, t):
    return x * np.cos(np.pi * t) + y * np.sin(np.pi * t)

t = np.arange(0, 1, 0.01)

for i in range(len(pos)):
    plt.plot(t, f(pos[i][0], pos[i][1], t)) 
'''
# 이미지를 입력받아서 경계선을 찾는것을 완료하라 

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




