# exp 3

# share factor
# gridnetwork_2wlane
# with traffic signals
import time
import resource 
from lib2to3.pgen2 import driver
from xml.dom.minicompat import NodeList
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
time_start = time.perf_counter()

dl = 200
beta_not_bike = 0.99
beta_not_car = 0.93
Go_2w = 209.37
Go_cars = 198.86
G1_2w = 326.75
G1_cars = 46.32
vsf_bike = 80
vsf_car = 100
x = np.linspace(0,5,num=10)
t = np.linspace(0,0.5,num = 100)
dx = 0.5
dt = 0.005
dts = dt * 3600
a = beta_not_bike
b = beta_not_car

ercar = []
erbike = []
## getting data from the excel file
ntwrk = pd.read_excel('grid3x3copy.xlsx', 'Sheet1')
odmatrix_bikes = pd.read_excel('grid3x3copy.xlsx', 'Sheet3')
odmatrix_cars = pd.read_excel('grid3x3copy.xlsx', 'Sheet4')
edgelabels = {}
row_ntwrk = ntwrk.iloc[:,0]
DG = nx.Graph()

## network construction
for row in range(0,np.size(row_ntwrk)):
    # by default it skips the first row (the headings)
    o = ntwrk.iloc[row,0]
    d = ntwrk.iloc[row,1]
    le = ntwrk.iloc[row,2]
    la = ntwrk.iloc[row,3]
    edgelabels[(o,d)] = la
    # appending the labels of links in the dictionary
    DG.add_edge(o,d,length=le,label=la)
    # constructing the network by iterations
    # all edge labels o,d: label
    # {(1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 5): 4, (5, 6): 5, (6, 7): 6, 
    # (7, 8): 7, (8, 9): 8, (9, 12): 13, (12, 11): 11, (11, 10): 10, 
    # (11, 17): 12, (17, 1): 14, (4, 16): 19, (5, 15): 18, (6, 14): 17, 
    # (7, 13): 16, (9, 1): 9, (3, 18): 15}
pos = nx.spring_layout(DG)
nx.draw(DG, pos, node_color='pink', with_labels=True)
nx.draw_networkx_edge_labels(DG, pos, edge_labels=edgelabels, label_pos=0.5)
plt.show() 
od_pairs_bikes = []
odpairs_bikes = {}
od_pairs_cars = []
odpairs_cars = {}
dt_car = []
dt_bike = []
ddt_car = []
ddt_bike = []

## storing origin destination data of bikes and cars
for orgn in range( 1, (len(odmatrix_bikes.iloc[:,0]) ) ):
    for destn in range( 1, (len(odmatrix_bikes.iloc[1,:]) ) ):
        org = int (odmatrix_bikes.iloc[orgn,0])
        des = int (odmatrix_bikes.iloc[0,destn])
        v = odmatrix_bikes.iloc[orgn,destn]
        odpairs_bikes[(org,des)] = v
        od_pairs_bikes.append(v)
    #print(od_pairs)
    #[127, 174, 186, 140, 160, 120, 148, 128, 137, 158, 164, 182, 126, 159, 122, 179]
    #print(odpairs)
    #{(10, 12): 127, (10, 15): 174, (10, 16): 186, (10, 17): 140, 
    #(13, 12): 160, (13, 15): 120, (13, 16): 148, (13, 17): 128, 
    #(14, 12): 137, (14, 15): 158, (14, 16): 164, (14, 17): 182, 
    #(18, 12): 126, (18, 15): 159, (18, 16): 122, (18, 17): 179}
    #print(odpairs[20,14])
    #to get the specific value for the key
for orgn in range( 1, (len(odmatrix_cars.iloc[:,0]) ) ):
    for destn in range( 1, (len(odmatrix_cars.iloc[1,:]) ) ):
        org = (odmatrix_cars.iloc[orgn,0])
        des = (odmatrix_cars.iloc[0,destn])
        #print(org, des)
        v = odmatrix_cars.iloc[orgn,destn]
        odpairs_cars[(org,des)] = v
        #print(orgn, destn,v)
        od_pairs_cars.append(v)

## function for finding the smallest element in the column
def smallestInCol(arr, row, colm):
    aa = 0
    for i in range(colm):         
        # initialize the minimum element as first element
        minm = arr[0][i] 
        # Run the inner loop for columns
        if i == m:
            for j in range(1, row):  
                # check if any element is smaller than the minimum element of the column and replace it
                if (arr[j][i] < minm):
                    minm = arr[j][i] 
                    aa = j
            break               
    return aa
    # gives the address of the lowest valued element in the list

## function for writing a csv file (4d kmatrix)
def writecsvkcar(k):
    with open('4kcar.csv', 'w') as outfile:
        # We write this header for readable, the pound symbol
        # will cause numpy to ignore it
        outfile.write('# Array shape: {0}\n'.format(k.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case.
        # Because we are dealing with 4D data instead of 3D data,
        # we need to add another for loop that's nested inside of the
        # previous one.
        a = -1
        for threeD_data_slice in k:
            for twoD_data_slice in threeD_data_slice:
                a = a + 1
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places. 
                np.savetxt(outfile, twoD_data_slice, fmt='%-8.2f')
                # Writing out a break to indicate different slices...
                outfile.write(str(a))
                outfile.write('# slice \n')

## function for calculating the density
def cal_densty(k_car, k_bike, k2w_bike, k2w_car,i,sr,d,ind,prev,edglen):
    for m in range(1,np.size(t)):
        k_bike[ind][i][len(edglen)-1][0] = 0
        k_car[ind][i][len(edglen)-1][0] = 0
        k2w_bike[ind][i][len(edglen)-1][0] = 0
        k2w_car[ind][i][len(edglen)-1][0] = 0
        if (sr==source):
            # link attached to the origin of O-D pair
            k_car[ind][i][0][m] = q_car[ind][m]/vsf_car
            k_bike[ind][i][0][m] = q_bike[ind][m]/vsf_bike
            k2w_bike[ind][i][0][m] = q_bike[ind][m]/vsf_bike
            k2w_car[ind][i][0][m] = q_car[ind][m]/vsf_car
            #print('kcar', k_car[ind][i][0][m], q_car[ind][m])
                # print(count, '81', ind, k_car[ind][0][0][0], q_car[ind], vsf_car)
            # for every first link in the path (the starting node is equal to 0, as here the od is 0 to 3)      
        else:
            if(prev!= -1):
                # the start node of a link, which is not a origin of O-D pair
                # the last but one term (space step density) from the previous link
                k_car[ind][i][0][m] = k_car[ind][prev][(len(edglen)-2)][m-1]
                k_bike[ind][i][0][m] = k_bike[ind][prev][(len(edglen)-2)][m-1]
                k2w_bike[ind][i][0][m] = k_bike[ind][prev][(len(edglen)-2)][m-1]
                k2w_car[ind][i][0][m] = k_car[ind][prev][(len(edglen)-2)][m-1]
                # in the following loop, (go to 3rd inner loop) prev gets the value of num from second iteration    

        if (sr == 2 and d == 3) or (sr == 1 and d == 2) or (sr == 1 and d == 4) or (sr == 4 and d == 5) or (sr == 7 and d == 8) or (sr == 4 and d == 7):
            for o in range(1,(len(edglen)-1)):  
                e = 0
                #f = k_bike[ind][i][o-1][m]
                g = 0
                h = k_car[ind][i][o-1][m-1]
                #u = k_car[ind][i][o-1][m]
                v = k_car[ind][i][o+1][m-1]
                #p = f+u
                r = v
                s = h

                if e == 0 and h == 0:
                    alf_2w_1 = 0
                else:
                    alf_2w_1 = e / s
                funct_1 = vsf_bike * e * a * (1 - ( s/(Go_2w+(G1_2w*alf_2w_1)) ))
                if funct_1 < 0:
                    funct_1=0

                if g == 0 and v == 0:
                    alf_2w_2 = 0
                else:
                    alf_2w_2 = g / r
                funct_2 = vsf_bike * g * a * (1 - ( r/(Go_2w+(G1_2w*alf_2w_2)) ))
                if funct_2 < 0:
                    funct_2 = 0

                if e == 0 and h == 0:
                    alf_car_1 = 0
                else:
                    alf_car_1 = e / s
                funct_3 = vsf_car * h * b * (1 - ( s/(Go_cars+(G1_cars*alf_car_1)) ))
                if funct_3 < 0:
                    funct_3 = 0

                if g == 0 and v == 0:
                    alf_car_2 = 0
                else:
                    alf_car_2 = g / r
                funct_4 = vsf_car * v * b * (1 - ( r/(Go_cars+(G1_cars*alf_car_2)) ))
                if funct_4 < 0:
                    funct_4 = 0

                k_bike[ind][i][o][m] = 0
                k_bike[ind][i][0][m] = 0
                k_car[ind][i][o][m] = (0.5*(h+v))-((0.5*dt/dx)*(funct_4-funct_3))

                # 8
                if sr == 5 and d == 8:
                    # blockage/ red time
                    if m < 15 or m in range(24, 45) or m > 56:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 7 and d == 8:
                    # blockage/ red time
                    if m < 27 or m in range(33, 57) or m > 62:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 9 and d == 8:
                    # blockage/ red time
                    if m < 36 or m in range(42, 66) or m > 71:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 5
                if sr == 2 and d == 5:
                    # blockage/ red time
                    if m < 15 or m in range(24, 57) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 5:
                    # blockage/ red time
                    if m < 36 or m in range(45, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 4 and d == 5:
                    # blockage/ red time
                    if m < 27 or m in range(33, 69) or m > 74:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 6 and d == 5:
                    # blockage/ red time
                    if m < 48 or m in range(54, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 2
                if sr == 5 and d == 2:
                    # blockage/ red time
                    if m < 15 or m in range(24, 57) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 3 and d == 2:
                    # blockage/ red time
                    if m < 27 or m in range(33, 57) or m > 62:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 1 and d == 2:
                    if m < 36 or m in range(42, 66) or m > 71:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 7
                if sr == 4 and d == 7:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 7:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 4
                if sr == 7 and d == 4:
                    if m < 15 or m in range(24, 48) or m > 56:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 1 and d == 4:
                    # blockage/ red time
                    if m < 36 or m in range(45, 69) or m > 77:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 5 and d == 4:
                    # blockage/ red time
                    if m < 27 or m in range(33, 60) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 1
                if sr == 4 and d == 1:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 2 and d == 1:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))  
                        # 9
                if sr == 6 and d == 9:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 9:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 6
                if sr == 3 and d == 6:
                    if m < 15 or m in range(24, 48) or m > 56:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 9 and d == 6:
                    # blockage/ red time
                    if m < 36 or m in range(45, 69) or m > 77:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 5 and d == 6:
                    # blockage/ red time
                    if m < 27 or m in range(33, 60) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                        # 3
                if sr == 6 and d == 3:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 2 and d == 3:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))



                k_bike[ind][i][len(edglen)-1][m] = 0
                k_car[ind][i][len(edglen)-1][m] = k_car[ind][i][len(edglen)-2][m]

            for o in range(1,(len(edglen)-1)):  
                e2w = k2w_bike[ind][i][o-1][m-1]
                #f = k_bike[ind][i][o-1][m]
                g2w = k2w_bike[ind][i][o+1][m-1]
                h2w = 0
                #u = k_car[ind][i][o-1][m]
                v2w = 0
                #p = f+u
                r2w = g2w
                s2w = e2w
                # print(o, e2w)

                if e == 0 and h == 0:
                    alf_2w_1 = 0
                else:
                    alf_2w_1 = e / s
                fa_12w = vsf_bike * e * a * (1 - ( s/(Go_2w+(G1_2w*alf_2w_1)) ))
                if fa_12w < 0:
                    fa_12w=0
                funct_12w = max(0, fa_12w)

                if g == 0 and v == 0:
                    alf_2w_2 = 0
                else:
                    alf_2w_2 = g / r
                fa_22w = vsf_bike * g * a * (1 - ( r/(Go_2w+(G1_2w*alf_2w_2)) ))
                if fa_22w < 0:
                    fa_22w = 0
                funct_22w = max(0, fa_22w)

                k2w_car[ind][i][o][m] = 0
                k2w_car[ind][i][0][m] = 0
                k2w_bike[ind][i][o][m] = (0.5*(e2w+g2w))-((0.5*dt/dx)*(funct_22w-funct_12w))

                # 8
                if sr == 5 and d == 8:
                    # blockage/ red time
                    if m < 15 or m in range(24, 45) or m > 56:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 7 and d == 8:
                    # blockage/ red time
                    if m < 27 or m in range(33, 57) or m > 62:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 9 and d == 8:
                    # blockage/ red time
                    if m < 36 or m in range(42, 66) or m > 71:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 5
                if sr == 2 and d == 5:
                    # blockage/ red time
                    if m < 15 or m in range(24, 57) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 5:
                    # blockage/ red time
                    if m < 36 or m in range(45, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 4 and d == 5:
                    # blockage/ red time
                    if m < 27 or m in range(33, 69) or m > 74:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 6 and d == 5:
                    # blockage/ red time
                    if m < 48 or m in range(54, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 2     
                if sr == 5 and d == 2:
                    # blockage/ red time
                    if m < 15 or m in range(24, 57) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 3 and d == 2:
                    # blockage/ red time
                    if m < 27 or m in range(33, 57) or m > 62:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 1 and d == 2:
                    if m < 36 or m in range(42, 66) or m > 71:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 7
                if sr == 4 and d == 7:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 7:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 4
                if sr == 7 and d == 4:
                    if m < 15 or m in range(24, 48) or m > 56:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 1 and d == 4:
                    # blockage/ red time
                    if m < 36 or m in range(45, 69) or m > 77:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 5 and d == 4:
                    # blockage/ red time
                    if m < 27 or m in range(33, 60) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 1
                if sr == 4 and d == 1:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 2 and d == 1:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 9
                if sr == 6 and d == 9:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 9:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 6
                if sr == 3 and d == 6:
                    if m < 15 or m in range(24, 48) or m > 56:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 9 and d == 6:
                    # blockage/ red time
                    if m < 36 or m in range(45, 69) or m > 77:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 5 and d == 6:
                    # blockage/ red time
                    if m < 27 or m in range(33, 60) or m > 65:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                        # 3
                if sr == 6 and d == 3:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 2 and d == 3:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k2w_bike[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k2w_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k2w_bike[ind][i][len(edglen)-3][m-1] == 0 and k2w_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k2w_car[ind][i][len(edglen)-3][m-1] / (k2w_bike[ind][i][len(edglen)-3][m-1] + k2w_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k2w_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k2w_bike[ind][i][len(edglen)-2][m] = (0.5*(k2w_bike[ind][i][len(edglen)-3][m-1]+k2w_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6- max(0,funct_5))))
                        k2w_car[ind][i][len(edglen)-2][m]  = (0.5*(k2w_car[ind][i][len(edglen)-3][m-1] +k2w_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))


                k2w_car[ind][i][len(edglen)-1][m] = 0
                k2w_bike[ind][i][len(edglen)-1][m] = k2w_bike[ind][i][len(edglen)-2][m]

                k_bike[ind][i][o][m] = k2w_bike[ind][i][o][m]
                k_bike[ind][i][0][m] = k2w_bike[ind][i][0][m]
                k_bike[ind][i][len(edglen)-1][m] = k2w_bike[ind][i][len(edglen)-1][m]

        else:
            for o in range(1,(len(edglen)-1)):  
                e = k_bike[ind][i][o-1][m-1]
                #f = k_bike[ind][i][o-1][m]
                g = k_bike[ind][i][o+1][m-1]
                h = k_car[ind][i][o-1][m-1]
                #u = k_car[ind][i][o-1][m]
                v = k_car[ind][i][o+1][m-1]
                #p = f+u
                r = g+v
                s = e+h

                if e == 0 and h == 0:
                    alf_2w_1 = 0
                else:
                    alf_2w_1 = e / s
                funct_1 = vsf_bike * e * a * (1 - ( s/(Go_2w+(G1_2w*alf_2w_1)) ))
                if funct_1 < 0:
                    funct_1=0

                if g == 0 and v == 0:
                    alf_2w_2 = 0
                else:
                    alf_2w_2 = g / r
                funct_2 = vsf_bike * g * a * (1 - ( r/(Go_2w+(G1_2w*alf_2w_2)) ))
                if funct_2 < 0:
                    funct_2 = 0

                if e == 0 and h == 0:
                    alf_car_1 = 0
                else:
                    alf_car_1 = e / s
                funct_3 = vsf_car * h * b * (1 - ( s/(Go_cars+(G1_cars*alf_car_1)) ))
                if funct_3 < 0:
                    funct_3 = 0

                if g == 0 and v == 0:
                    alf_car_2 = 0
                else:
                    alf_car_2 = g / r
                funct_4 = vsf_car * v * b * (1 - ( r/(Go_cars+(G1_cars*alf_car_2)) ))
                if funct_4 < 0:
                    funct_4 = 0

                k_bike[ind][i][o][m] = (0.5*(e+g))-((0.5*dt/dx)*(funct_2-funct_1))
                k_car[ind][i][o][m] = (0.5*(h+v))-((0.5*dt/dx)*(funct_4-funct_3))

                # 8
                if sr == 5 and d == 8:
                    # blockage/ red time
                    if m < 15 or m in range(24, 45) or m > 56:

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0

                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 7 and d == 8:
                    # blockage/ red time
                    if m < 27 or m in range(33, 57) or m > 62:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 9 and d == 8:
                    # blockage/ red time
                    if m < 36 or m in range(42, 66) or m > 71:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 5
                if sr == 2 and d == 5:
                    # blockage/ red time
                    if m < 15 or m in range(24, 57) or m > 65:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 5:
                    # blockage/ red time
                    if m < 36 or m in range(45, 78) or m > 86:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 4 and d == 5:
                    # blockage/ red time
                    if m < 27 or m in range(33, 69) or m > 74:

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 6 and d == 5:
                    # blockage/ red time
                    if m < 48 or m in range(54, 90) or m > 95:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 2
                if sr == 5 and d == 2:
                    # blockage/ red time
                    if m < 15 or m in range(24, 57) or m > 65:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 3 and d == 2:
                    # blockage/ red time
                    if m < 27 or m in range(33, 57) or m > 62:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 1 and d == 2:
                    if m < 36 or m in range(42, 66) or m > 71:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 7
                if sr == 4 and d == 7:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 7:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 4
                if sr == 7 and d == 4:
                    if m < 15 or m in range(24, 48) or m > 56:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 1 and d == 4:
                    # blockage/ red time
                    if m < 36 or m in range(45, 69) or m > 77:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 5 and d == 4:
                    # blockage/ red time
                    if m < 27 or m in range(33, 60) or m > 65:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 1
                if sr == 4 and d == 1:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 2 and d == 1:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 9
                if sr == 6 and d == 9:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 8 and d == 9:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 6
                if sr == 3 and d == 6:
                    if m < 15 or m in range(24, 48) or m > 56:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 9 and d == 6:
                    # blockage/ red time
                    if m < 36 or m in range(45, 69) or m > 77:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 5 and d == 6:
                    # blockage/ red time
                    if m < 27 or m in range(33, 60) or m > 65:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
                        # 3
                if sr == 6 and d == 3:
                    if m < 15 or m in range(24, 36) or m in range(45, 57) or m in range(66, 78) or m > 86:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

                if sr == 2 and d == 3:
                    if m < 27 or m in range(33, 48) or m in range(54, 69) or m in range(75, 90) or m > 95:
                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_2w_1 = 0
                        else:
                            alfa_2w_1 = k_bike[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_5 = vsf_bike * k_bike[ind][i][len(edglen)-3][m-1] * a * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_2w + (G1_2w*alfa_2w_1)) ))
                        if funct_5 < 0:
                            funct_5 = 0
                        funct_6 = 0

                        if k_bike[ind][i][len(edglen)-3][m-1] == 0 and k_car[ind][i][len(edglen)-3][m-1] == 0:
                            alfa_car_1 = 0
                        else:
                            alfa_car_1 = k_car[ind][i][len(edglen)-3][m-1] / (k_bike[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1])
                        funct_7 = vsf_car * k_car[ind][i][len(edglen)-3][m-1] * b * (1 - ( (k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1]) / (Go_cars+(G1_cars*alfa_car_1)) ))
                        if funct_7 < 0:
                            funct_7 = 0
                        funct_8 = 0
                        
                        k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-1][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                        k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-1][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))


                k_bike[ind][i][len(edglen)-1][m] = k_bike[ind][i][len(edglen)-2][m]
                k_car[ind][i][len(edglen)-1][m] = k_car[ind][i][len(edglen)-2][m]

# velocity and insta travel time
    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x))):
            # velocity matrix
            if k_car[ind][i][p][m] < 0.1:
                v_car[ind][i][p][m] = vsf_car
            else:
                #print(v1_car[p][m], q1_car[m], k1_car[p][m])
                if k_car[ind][i][p][m] == 0 and k_bike[ind][i][p][m] == 0:
                    alp_car = 0
                else:
                    alp_car = k_car[ind][i][p][m] / ( k_bike[ind][i][p][m] + k_car[ind][i][p][m])
                v_car[ind][i][p][m] = vsf_car * b * (1 - ( k_car[ind][i][p][m] + k_bike[ind][i][p][m] ) / (Go_cars+(G1_cars*alp_car)))
            if k_bike[ind][i][p][m] < 0.1:
                v_bike[ind][i][p][m] = vsf_bike
            else:
                if k_car[ind][i][p][m] == 0 and k_bike[ind][i][p][m] == 0:
                    alp_2w = 0
                else:
                    alp_2w = k_bike[ind][i][p][m] / ( k_bike[ind][i][p][m] + k_car[ind][i][p][m])
                v_bike[ind][i][p][m] = vsf_bike * a * (1 - ( k_car[ind][i][p][m] + k_bike[ind][i][p][m] ) / (Go_2w+(G1_2w*alp_2w)))

            if v_car[ind][i][p][m] > 100:
                v_car[ind][i][p][m] = vsf_car
            if v_bike[ind][i][p][m] > 80:
                v_bike[ind][i][p][m] = vsf_bike
            if v_car[ind][i][p][m] < 0.1:
                v_car[ind][i][p][m] = 0.1
            if v_bike[ind][i][p][m] < 0.1:
                v_bike[ind][i][p][m] = 0.1
            #print(v_car[ind][i][p][m], k_car[ind][i][p][m])

            tt_car[ind][i][p][m] = (dx)*3600/v_car[ind][i][p][m]
            tt_bike[ind][i][p][m] = (dx)*3600/v_bike[ind][i][p][m]
    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x))):
            if v_car[ind][i][p][m] < 5:
                tt_car[ind][i][p][m] = 54
            if v_bike[ind][i][p][m] < 5:
                tt_bike[ind][i][p][m] = 54

def contrplt(r,aac, titl):
    t = np.linspace(0,0.5,num = 100)
    s = np.linspace(0,5,num= 10)
    xgrid = np.linspace(0, 30, 100)
    ygrid = np.linspace(0, 5, 10)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = r
    ax = fig.add_subplot(aac)
    ax.set_xlabel('departure time (min)')
    ax.set_ylabel('road stretch (km)')
    plt.imshow(Z.reshape(Xgrid.shape),
                origin='lower', aspect='auto',
                extent=[0, 30, 0, 5],
                cmap='OrRd')
    plt.clim(0,250)
    cb = plt.colorbar()
    cb.set_label('Density (veh/km)')    

def trav_tim(x1,a,col):
    trv_tim = []
    tot_tim = 0
    for i in range(0, np.size(x1)):
        for j in range(0, 1):
            if col >= np.size(t):
                break
            tt = int(a[ind][num][i][col-1])
            col = int(np.round( ((tt / dts) + col), 0) )
            trv_tim.append(tt)
            if tt >= np.size(t) or col >= np.size(t):
                break
    if len(trv_tim) < np.size(x1):
        trv_tim = np.zeros([np.size(t)])
    for i in range(0, len(trv_tim)):
        tot_tim = trv_tim[i] + tot_tim
    return tot_tim

def d_trav_tim(x1,a,col):
    trv_tim = []
    tot_tim = 0
    for i in range(0, np.size(x1)):
        for j in range(0, 1):
            if col >= np.size(t):
                break
            tt = int(a[ind][num][i][col-1])
            col = int(np.round( ((tt / dts) + col), 0) )
            trv_tim.append(tt)
            if tt >= np.size(t) or col >= np.size(t):
                break
    for i in range(0, len(trv_tim)):
        tot_tim = trv_tim[i] + tot_tim
    return tot_tim

def elim(a):
    for i in range(0, np.size(a)):
        for j in range(0, np.size(a)):
            b = np.size(a)
            if a[b-j-1] == 0 and (b-j-1) > 0:
                #print(np.size(a), (b-j-1))
                a.pop(np.size(a)-j-1)
                #print(a)
    return a

## storing the dictionary data to dcar variable
for orgn in range( 1, (len(odmatrix_bikes.iloc[:,0]) ) ):
    for destn in range( 1, (len(odmatrix_bikes.iloc[1,:]) ) ):
        print(" --------------------------------------------------------------------------------------------------------------------------------------------------------------------- ")
        print(" --------------------------------------------------------------------------------------------------------------------------------------------------------------------- ")
        source = int (odmatrix_bikes.iloc[orgn,0])
        destination = int (odmatrix_bikes.iloc[0,destn])
        d_bike = odpairs_bikes[source, destination]
        d_car = odpairs_cars[source, destination]
        if d_bike == 0 or d_car == 0:
            break
        # demand entering per dt (number of vehicles)
        paths = nx.all_simple_edge_paths(DG, source, destination)
        p = (list(paths))

        print(p)
        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(paths)
        print(len(p))
        # gives the number of paths from the particular source and destination

        q_car = np.zeros([np.size(p), np.size(t)])
        q_bike = np.zeros([np.size(p), np.size(t)])
        length=[0]*len(p)
        err_car = np.zeros([np.size(t)])
        err_bike = np.zeros([np.size(t)])
        k_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        k_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        k2w_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        k2w_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        v_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        v_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        v2w_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        tt_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        tt_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        tt2w_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
        dtrav_car = np.zeros((len(p), len(DG.edges)+1, np.size(t)))
        dtrav_bike = np.zeros((len(p), len(DG.edges)+1, np.size(t)))
        dtrav2w_bike = np.zeros((len(p), len(DG.edges)+1, np.size(t)))
        dtravp_car = np.zeros([len(p), np.size(t)])
        dtravp_bike = np.zeros([len(p), np.size(t)])
        dtravp2w_bike = np.zeros([len(p), np.size(t)])

## MSA algorithm application for the chosen O-D pair from the network
        count=2
        while True:
## path chosen from the possible paths avaliable from chosen O-D pair
            for path in p:
                temp3 = 0
                prev=-1
                ind = p.index(path)
                #print(ind)
                if count==2:
                    for m in range(0, np.size(t)):
                        q_car[ind][m] = d_car/np.size(p)
                        q_bike[ind][m] = d_bike/np.size(p)
                    
## link in the path chosen, density matrix calculation
                for edge in path:
                    #print('path', path, ind, edge)
                    data = DG.get_edge_data(edge[0],edge[1])
                    # edge with start node and end node data
                    num = data['label']
                    temp3+= data['length']
                    edglen = np.linspace( 0, data['length'], num = int((data['length'])/dx) )
                    # stores length of the edge
                    srt = edge[0]
                    dst = edge[1]
                    #print('data, num, temp3', data, num, temp3, prev, srt, dst)
                    cal_densty(k_car, k_bike, k2w_bike, k2w_car,num,srt,dst,ind,prev,edglen)
                    for i in range(1, np.size(t)+1):
                        dt_bike.append(trav_tim(x,tt_bike, i))
                        ddt_bike.append(d_trav_tim(x,tt_bike, i))
                        dt_car.append(trav_tim(x,tt_car, i))
                        ddt_car.append(d_trav_tim(x,tt_car, i))
                    writecsvkcar(k_car)
                    #print(ind, num, ddt_bike, dt_bike)
                    dtrav_bike[ind][num][:] = ddt_bike
                    dtrav_car[ind][num][:] = ddt_car
                    #print("dtrav_bikedtrav_bike", dtrav_bike)

                    ddt_bike = []
                    ddt_car = []
                    # num stores index of the link
                    # value of num is pushed to prev
                    prev = num
                    # the first and last links of the path e1 AND e2
                    if(srt==source):
                        e1 = DG.edges[srt,edge[1]]['label']
                        # e1 is an edge starting from node 0 
                        # (either (0,1) or (0,2); 0 and 1 respectievely values for e1)
                    elif(dst==destination):
                        e2 = DG.edges[edge[0],dst]['label']
                        # e2 is an edge ending with node dst
                    else:
                        continue   

## velocity , instantaneous travel time and dynamic travel time matrices
                for m in range(1,np.size(t)):
                        #, 3 indices so 3 elements in the list
                    length[ind] = temp3


            for path in p:
                ind = p.index(path)
                for i in range(0, np.size(t)):
                    for edge in path:
                        data = DG.get_edge_data(edge[0],edge[1])
                        num = data['label']
                        #print(p, ind, i, num)
                        dtravp_bike[ind][i] = dtravp_bike[ind][i] + int(dtrav_bike[ind][num][i])
                        dtravp_car[ind][i] = dtravp_car[ind][i] + int(dtrav_car[ind][num][i])

            # plotting the dynamic travel times
            if count == 3 or count == 150:
                clrs = ["blue", "orange","green","red","purple","brown","pink","gray","olive","cyan","black"]
                fig = plt.figure(figsize=(8, 6), dpi=80)
                tx1 = fig.add_subplot(221)
                for i in range(0, len(p)):
                    plt.plot(range(0,np.size(dtravp_bike[i][:])),dtravp_bike[i][:],color=clrs[i])
                tx1.set(title = '', xlabel = 'departure time', ylabel = 'travel time of bikes (sec)')
                tx1.set_xlim([0, 75])
                tx1.set_ylim([0, 3500])

                tx2 = fig.add_subplot(222)
                for i in range(0, len(p)):
                    plt.plot(range(0,np.size(dtravp_car[i][:])),dtravp_car[i][:],color=clrs[i], label = i)

                tx2.set_xlim([0, 75])
                tx2.set_ylim([0, 3500])
                tx2.set(title = '', xlabel = 'departure time', ylabel = 'travel time of cars (sec)')
                plt.legend()
                plt.show()
                break

## storing the error 
            temp1 = np.transpose(dtravp_bike)
            temp1_car = np.transpose(dtravp_car)
            for m in range(0, np.size(t)): 
                #print("                print(dtravp_car)", dtravp_car)
                nmrtr_car = 0
                dnmtr_car = 0             
                nmrtr_bike = 0
                dnmtr_bike = 0
                for path in p:
                    ind = p.index(path)
                    nmrtr_car = nmrtr_car + ( dtravp_car[ind][m] * q_car[ind][m] )
                    nmrtr_bike = nmrtr_bike + ( dtravp_bike[ind][m] * q_bike[ind][m] )
                dnmtr_car = min(temp1_car[m][:]) * d_car
                dnmtr_bike = min(temp1[m][:]) * d_bike
                #print("nmrtr", nmrtr_bike, dnmtr_bike)

                err_car[m] = (nmrtr_car/dnmtr_car) - 1                
                err_bike[m] = (nmrtr_bike/dnmtr_bike) - 1

                #if m == 99:
                    #print(min1, t_car)
                    #print(q_car)

            #if count == 2:
                #print(count,(k_car[0][:][:][:]))
                #print('t_bike', t_bike[1][:])
                #print('errbike', err_bike)
                #print('q_bike', q_bike)
            #print('t_bike', t_bike[0][:])
            #print('t_bike', t_bike[2][:])

            error_car = max(err_car)
            error_bike = max(err_bike)
            ercar.append(error_car)
            erbike.append(error_bike)
            ercarx = np.size(ercar)

            if count == 1000 or count == 120 or (error_car < 0.002 and error_bike < 0.002):
                fig = plt.figure()
                erc = ercar[:]
                erb = erbike[:]
                qx1 = fig.add_subplot(221)
                plt.plot(range(0,np.size(erc)),erc,color="blue", label = 'car')
                plt.plot(range(0,np.size(erb)),erb,color="orange", label = 'bike')
                qx1.set_xlim([0, 200])
                qx1.set(title = 'relative gap',
                        xlabel = 'iterations',
                        ylabel = 'error')
                plt.legend()
                plt.show() 
            if error_car < 0.002 and error_bike < 0.002:
                fig = plt.figure()
                tx1 = fig.add_subplot(221)
                plt.plot(range(0,np.size(dtravp_bike[0][:])),dtravp_bike[0][:],color="midnightblue")
                plt.plot(range(0,np.size(dtravp_bike[1][:])),dtravp_bike[1][:],color="darkblue")
                plt.plot(range(0,np.size(dtravp_bike[2][:])),dtravp_bike[2][:],color="blue")
                plt.plot(range(0,np.size(dtravp_bike[3][:])),dtravp_bike[3][:],color="slateblue")
                plt.plot(range(0,np.size(dtravp_bike[4][:])),dtravp_bike[4][:],color="darkslateblue")
                plt.plot(range(0,np.size(dtravp_bike[5][:])),dtravp_bike[5][:],color="mediumslateblue")
                plt.plot(range(0,np.size(dtravp_bike[6][:])),dtravp_bike[6][:],color="blueviolet")
                plt.plot(range(0,np.size(dtravp_bike[7][:])),dtravp_bike[7][:],color="plum")
                plt.plot(range(0,np.size(dtravp_bike[8][:])),dtravp_bike[8][:],color="violet")
                plt.plot(range(0,np.size(dtravp_bike[9][:])),dtravp_bike[9][:],color="thistle")
                plt.plot(range(0,np.size(dtravp_bike[10][:])),dtravp_bike[10][:],color="magenta")
                tx1.set(title = 'travel time', xlabel = 'simulation time', ylabel = 'bike')
                tx1.set_xlim([0, 100])

                tx2 = fig.add_subplot(222)
                plt.plot(range(0,np.size(dtravp_car[0][:])),dtravp_car[0][:],color="rosybrown", label = '0')
                plt.plot(range(0,np.size(dtravp_car[1][:])),dtravp_car[1][:],color="lightcoral", label = '1')
                plt.plot(range(0,np.size(dtravp_car[2][:])),dtravp_car[2][:],color="indianred", label = '2')
                plt.plot(range(0,np.size(dtravp_car[3][:])),dtravp_car[3][:],color="brown", label = '3')
                plt.plot(range(0,np.size(dtravp_car[4][:])),dtravp_car[4][:],color="maroon", label = '4')
                plt.plot(range(0,np.size(dtravp_car[5][:])),dtravp_car[5][:],color="salmon", label = '5')
                plt.plot(range(0,np.size(dtravp_car[6][:])),dtravp_car[6][:],color="orangered", label = '6')
                plt.plot(range(0,np.size(dtravp_car[7][:])),dtravp_car[7][:],color="sienna", label = '7')
                plt.plot(range(0,np.size(dtravp_car[8][:])),dtravp_car[8][:],color="peru", label = '8')
                plt.plot(range(0,np.size(dtravp_car[9][:])),dtravp_car[9][:],color="sandybrown", label = '9')
                plt.plot(range(0,np.size(dtravp_car[10][:])),dtravp_car[10][:],color="peachpuff", label = '10')
                tx2.set_xlim([0, 100])
                tx2.set(title = count, xlabel = 'simulation time', ylabel = 'car')
                plt.legend()
                plt.show()

                break
            # print(error_car)
            #print(count)
            #print(t_car)
## redistribution of flow
            for m in range(0, np.size(t)):
                temp2_car = min(temp1_car[:][m])
                temp3_car = np.where(temp1_car[:][m] == temp2_car)
                temp4_car = temp3_car[0]
                min1 = temp4_car[0]                
                q_car[min1][m] = q_car[min1][m]*((count-2)/(count-1))+(d_car/(count-1))
                for i in range(len(p)):
                    if(i!=min1):
                        q_car[i][m] = q_car[i][m]*((count-2)/(count-1)) 

                temp2 = min(temp1[:][m])
                temp3 = np.where(temp1[:][m] == temp2)
                temp4 = temp3[0]
                min2 = temp4[0]
                #print(m, "temp2, min1",  temp2, min1, temp3)
                q_bike[min2][m] = q_bike[min2][m]*((count-2)/(count-1))+(d_bike/(count-1))
                for i in range(len(p)):
                    if(i!=min2):
                        q_bike[i][m] =q_bike[i][m]*((count-2)/(count-1))
            #print("  qcar  ", q_car)
            #print("  qbike  ", q_bike)


                #print("print(q_car, q_bike)", q_car, q_bike, d_car, d_bike)
            #if (error_car <=0.002):
                #break
## stopping criteria

            count+=1
            #print('t_bike',t_bike)

            k_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            k_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            k2w_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            k2w_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            v_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            v_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            tt_car = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            tt_bike = np.zeros((len(p),len(DG.edges)+1,np.size(x),np.size(t)))
            dtrav_car = np.zeros((len(p), len(DG.edges)+1, np.size(t)))
            dtrav_bike = np.zeros((len(p), len(DG.edges)+1, np.size(t)))
            dtravp_bike = np.zeros([len(p), np.size(t)])
            dtravp_car = np.zeros([len(p), np.size(t)])
            length=[0]*len(p)
            err_car = np.zeros([np.size(t)])
            err_bike = np.zeros([np.size(t)])

            for i in range(0, np.size(x)):
                k_car[0][0][i][0] = 0.15
                k_bike[0][0][i][0] = 0.1875

            print('count',count)
            #print('q_bike', q_bike)

            if count == 101 or (error_car < 0.002 and error_bike < 0.002):
                break

# travel time
# calibration 
# density 
time_elapsed = (time.perf_counter() - time_start)
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ( "------======-------" ,  "%5.1f secs %5.1f MByte" % (time_elapsed,memMb),  "------======-------" )