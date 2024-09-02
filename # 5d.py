# 5d

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
b0b = 0.98
a = b0b
b0c = 0.94
b = b0c
b1b = 0.52
c = b1b
b1c = 0.86
dd = b1c
vsf_bike = 80
vsf_car = 100
x = np.linspace(0,5,num=10)
t = np.linspace(0,0.5,num = 100)
dx = 0.5
dt = 0.005
dts = dt * 3600

ercar = []
erbike = []
avg_trv = []
avg_trv_car = []
avg_trv_bike = []
overal_sys_trv_timb = 0
overal_sys_trv_timc = 0
overal_sys_trv_tim = 0
## getting data from the excel file
ntwrk = pd.read_excel('exp2.xlsx', 'Sheet1')
odmatrix_bikes = pd.read_excel('exp2.xlsx', 'Sheet3')
odmatrix_cars = pd.read_excel('exp2.xlsx', 'Sheet4')
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
#plt.show() 
od_pairs_bikes = []
odpairs_bikes = {}
od_pairs_cars = []
odpairs_cars = {}
dt_car = []
dt_bike = []
ddt_car = []
ddt_bike = []
od_bike = {}

lnks = [0,1,2,3,4,5]
od_temp = 0
d_car = []
d_bike = []
## storing origin destination data of bikes and cars
for orgn in range( 1, (len(odmatrix_bikes.iloc[:,0]) ) ):
    for destn in range( 1, (len(odmatrix_bikes.iloc[1,:]) ) ):
        od_temp = od_temp + 1
        org = int (odmatrix_bikes.iloc[orgn,0])
        des = int (odmatrix_bikes.iloc[0,destn])
        v = odmatrix_bikes.iloc[orgn,destn]
        odpairs_bikes[(org,des)] = v        # dict
        od_pairs_bikes.append(v)            # list    
        od_bike[od_temp] = (org, des)       # dict 1,2,3...
        d_bike.append(v)
print(od_bike)
print(d_bike)

p = []
for orgn in range( 1, (len(odmatrix_bikes.iloc[:,0]) ) ):
    for destn in range( 1, (len(odmatrix_bikes.iloc[1,:]) ) ):
        source = int (odmatrix_bikes.iloc[orgn,0])
        destination = int (odmatrix_bikes.iloc[0,destn])
        # demand entering per dt (number of vehicles)
        paths = nx.all_simple_edge_paths(DG, source, destination)
        p1 = (list(paths))
        p.append(p1)
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(p[0], np.size(p[0]))
pp = []
for i in range(0, len(p)):
    abcd = p[i]
    for j in range(0, len(p[i])):
        pp.append(abcd[j])
print("::::::::::::", np.size(pp))
lmnoo = pp[0]
print(np.size(pp[1]))
print(pp)
# for j in range(0, np.size(pp)):
#   for i in range(0, (np.size(pp[j])/2) ):
#       (  ((np.size(pp[j])) / 2) - 8 + i  )
# index of the link in the route
print(lmnoo[1])
print(pp[1])

for orgn in range( 1, (len(odmatrix_cars.iloc[:,0]) ) ):
    for destn in range( 1, (len(odmatrix_cars.iloc[1,:]) ) ):
        org = int (odmatrix_cars.iloc[orgn,0])
        des = int (odmatrix_cars.iloc[0,destn])
        # print(org, des)
        v = odmatrix_cars.iloc[orgn,destn]
        odpairs_cars[(org,des)] = v
        #print(orgn, destn,v)
        od_pairs_cars.append(v)
        d_car.append(v)

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
            tt = int(a[ind][nu][i][col-1])
            col = int(np.round( ((tt / dts) + col), 0) )
            trv_tim.append(tt)
            if tt >= np.size(t) or col >= np.size(t):
                break
    for i in range(0, len(trv_tim)):
        tot_tim = trv_tim[i] + tot_tim
    return tot_tim

## function for calculating the density
def cal_densty(k_car, k_bike, i,sr,d,ind,prev,edglen):
    for m in range(1,np.size(t)):
        k_bike[ind][i][len(edglen)-1][0] = 0
        k_car[ind][i][len(edglen)-1][0] = 0
        if (sr==source):
            # link attached to the origin of O-D pair
            k_car[ind][i][0][m] = q_car[ind][m]/vsf_car
            k_bike[ind][i][0][m] = q_bike[ind][m]/vsf_bike
            #print('kcar', k_car[ind][i][0][m], q_car[ind][m])
                # print(count, '81', ind, k_car[ind][0][0][0], q_car[ind], vsf_car)
            # for every first link in the path (the starting node is equal to 0, as here the od is 0 to 3)      
        else:
            if(prev!= -1):
                # the start node of a link, which is not a origin of O-D pair
                # the last but one term (space step density) from the previous link
                k_car[ind][i][0][m] = k_car[ind][prev][(len(edglen)-2)][m-1]
                k_bike[ind][i][0][m] = k_bike[ind][prev][(len(edglen)-2)][m-1]
                # in the following loop, (go to 3rd inner loop) prev gets the value of num from second iteration    
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
            funct_1 = min( (vsf_bike/dl)*(e*(b0b * dl - b1b * s)), 3476)

            if funct_1 < 0:
                funct_1 = 0
            funct_2 = min( (vsf_bike/dl)*(g*(b0b * dl - b1b * r)), 3476)
            if funct_2 < 0:
                funct_2 = 0
            funct_3 = min( (vsf_car/dl)*(h*(b0c * dl - b1c * s)), 1942)
            if funct_3 < 0:
                funct_3 = 0            
            funct_4 = min( (vsf_car/dl)*(v*(b0c * dl - b1c * r)), 1942)
            if funct_4 < 0:
                funct_4 = 0

            k_bike[ind][i][o][m] = (0.5*(e+g))-((0.5*dt/dx)*(funct_2 - funct_1))
            k_car[ind][i][o][m] = (0.5*(h+v))-((0.5*dt/dx)*(funct_4 - funct_3))

            # 2
            if sr == 1 and d == 2:
                # blockage/ red time
                if m < 15 or m in range(24, 43) or m in range(52,69) or m > 77:
                    funct_5 = (vsf_bike/dl)*(k_bike[ind][i][len(edglen)-3][m-1]*(a*dl-c*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    #print("funct_5", funct_5, np.size(funct_5))
                    if (funct_5<0):
                        funct_5=0
                    funct_6 = 0
                    funct_7 = (vsf_car/dl)*(k_car[ind][i][len(edglen)-3][m-1]*(b*dl-dd*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    if (funct_7<0):
                        funct_7=0
                    funct_8 = 0
                    k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1] + k_bike[ind][i][len(edglen)-3][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                    k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] + k_car[ind][i][len(edglen)-3][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))


            if sr == 3 and d == 2:
                # blockage/ red time
                if m < 24 or m in range(33, 52) or m in range(61,78) or m > 86:
                    funct_5 = (vsf_bike/dl)*(k_bike[ind][i][len(edglen)-3][m-1]*(a*dl-c*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    #print("funct_5", funct_5, np.size(funct_5))
                    if (funct_5<0):
                        funct_5=0
                    funct_6 = 0
                    funct_7 = (vsf_car/dl)*(k_car[ind][i][len(edglen)-3][m-1]*(b*dl-dd*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    if (funct_7<0):
                        funct_7=0
                    funct_8 = 0
                    k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-3][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                    k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-3][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))


            if sr == 4 and d == 2:
                # blockage/ red time
                if m < 33 or m in range(42, 61) or m > 69:
                    funct_5 = (vsf_bike/dl)*(k_bike[ind][i][len(edglen)-3][m-1]*(a*dl-c*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    #print("funct_5", funct_5, np.size(funct_5))
                    if (funct_5<0):
                        funct_5=0
                    funct_6 = 0
                    funct_7 = (vsf_car/dl)*(k_car[ind][i][len(edglen)-3][m-1]*(b*dl-dd*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    if (funct_7<0):
                        funct_7=0
                    funct_8 = 0
                    k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-3][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                    k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-3][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))
               
                    # 3
            if sr == 1 and d == 3:
                # blockage/ red time
                if m < 15 or m in range(21, 33) or m in range(39, 50) or m in range(56, 68) or m in range(74, 86) or m > 91:
                    funct_5 = (vsf_bike/dl)*(k_bike[ind][i][len(edglen)-3][m-1]*(a*dl-c*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    #print("funct_5", funct_5, np.size(funct_5))
                    if (funct_5<0):
                        funct_5=0
                    funct_6 = 0
                    funct_7 = (vsf_car/dl)*(k_car[ind][i][len(edglen)-3][m-1]*(b*dl-dd*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    if (funct_7<0):
                        funct_7=0
                    funct_8 = 0
                    k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-3][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                    k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-3][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

            if sr == 2 and d == 3:
                # blockage/ red time
                if m < 21 or m in range(27, 39) or m in range(45, 56) or m in range(62, 74) or m in range(80, 92) or m > 97:
                    funct_5 = (vsf_bike/dl)*(k_bike[ind][i][len(edglen)-3][m-1]*(a*dl-c*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    #print("funct_5", funct_5, np.size(funct_5))
                    if (funct_5<0):
                        funct_5=0
                    funct_6 = 0
                    funct_7 = (vsf_car/dl)*(k_car[ind][i][len(edglen)-3][m-1]*(b*dl-dd*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    if (funct_7<0):
                        funct_7=0
                    funct_8 = 0
                    k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-3][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                    k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-3][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))

            if sr == 4 and d == 3:
                # blockage/ red time
                if m < 27 or m in range(33, 45) or m in range(51, 62) or m in range(68, 80) or m > 85:
                    funct_5 = (vsf_bike/dl)*(k_bike[ind][i][len(edglen)-3][m-1]*(a*dl-c*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    #print("funct_5", funct_5, np.size(funct_5))
                    if (funct_5<0):
                        funct_5=0
                    funct_6 = 0
                    funct_7 = (vsf_car/dl)*(k_car[ind][i][len(edglen)-3][m-1]*(b*dl-dd*(k_bike[ind][i][len(edglen)-3][m-1]+k_car[ind][i][len(edglen)-3][m-1])))
                    if (funct_7<0):
                        funct_7=0
                    funct_8 = 0
                    k_bike[ind][i][len(edglen)-2][m] = (0.5*(k_bike[ind][i][len(edglen)-3][m-1]+k_bike[ind][i][len(edglen)-3][m-1])-((0.5*dt/dx)*(funct_6-funct_5)))
                    k_car[ind][i][len(edglen)-2][m]  = (0.5*(k_car[ind][i][len(edglen)-3][m-1] +k_car[ind][i][len(edglen)-3][m-1]) -((0.5*dt/dx)*(funct_8-funct_7)))


            k_bike[ind][i][len(edglen)-1][m] = k_bike[ind][i][len(edglen)-2][m]
            k_car[ind][i][len(edglen)-1][m] = k_car[ind][i][len(edglen)-2][m]



            k_bike[ind][i][len(edglen)-1][m] = k_bike[ind][i][len(edglen)-2][m]
            k_car[ind][i][len(edglen)-1][m] = k_car[ind][i][len(edglen)-2][m]

    # adding up the links flow
    if ind == (np.size(pp) - 1) and i == 4:
        print(ind, pp[ind])
        for link in lnks:
            for route in range(0,np.size(pp)):
                for tim in range(0, np.size(t)):
                    for sps in range(0, np.size(x)):
                        if route + 1 == np.size(pp):
                            break
                        k_bike[route+1][link][sps][tim] = k_bike[route+1][link][sps][tim] + k_bike[route][link][sps][tim]
                        k_car[route+1][link][sps][tim] = k_car[route+1][link][sps][tim] + k_car[route][link][sps][tim]

        for link in lnks:
            for route in range(0,np.size(pp)):
                for tim in range(0, np.size(t)):
                    for sps in range(0, np.size(x)):
                        k_car[route][link][sps][tim] = k_car[(np.size(pp)-1)][link][sps][tim]
                        k_bike[route][link][sps][tim] = k_bike[(np.size(pp)-1)][link][sps][tim]


            # velocity matrix
        for ind in range(0,np.size(pp)):
            for i in lnks:
                for m in range(0, np.size(t)):    
                    for p in range(0, (np.size(x))):
                        if k_car[ind][i][p][m] < 0.1:
                            v_car[ind][i][p][m] = vsf_car
                        else:
                            v_car[ind][i][p][m] = vsf_car * ( b0c * dl  - b1c * ( k_car[ind][i][p][m] + k_bike[ind][i][p][m] ) ) / dl
                        if k_bike[ind][i][p][m] < 0.1:
                            v_bike[ind][i][p][m] = vsf_bike
                        else:
                            v_bike[ind][i][p][m] = vsf_bike * ( b0b * dl  - b1b * ( k_car[ind][i][p][m] + k_bike[ind][i][p][m] ) ) / dl
                        if v_car[ind][i][p][m] > 100:
                            v_car[ind][i][p][m] = vsf_car
                        if v_bike[ind][i][p][m] > 80:
                            v_bike[ind][i][p][m] = vsf_bike
                        if v_car[ind][i][p][m] < 0.1:
                            v_car[ind][i][p][m] = 0.1
                        if v_bike[ind][i][p][m] < 0.1:
                            v_bike[ind][i][p][m] = 0.1

                        tt_car[ind][i][p][m] = (dx)*3600/v_car[ind][i][p][m]
                        tt_bike[ind][i][p][m] = (dx)*3600/v_bike[ind][i][p][m]
        for ind in range(0,np.size(pp)):
            for i in lnks:
                for m in range(0, np.size(t)):    
                    for p in range(0, (np.size(x))):
                        if v_car[ind][i][p][m] < 5:
                            tt_car[ind][i][p][m] = 54
                        if v_bike[ind][i][p][m] < 5:
                            tt_bike[ind][i][p][m] = 54



def contrplt(r,aac, titl):
    t = np.linspace(0,0.5,num = 100)
    s = np.linspace(0,5,num= 10)
    [tx, sy] = np.meshgrid(t,s)
    ax = fig.add_subplot(aac)
    ax.contourf(t,s,r)
    ax.set_title(titl)
    ax.set_xlabel('simulation time')
    ax.set_ylabel('road stretch')
    #cs.cmap.set_over('red')
    plt.imshow(r, extent=[0, 120, 0, 100], origin='lower',
            cmap='OrRd')
    plt.clim(0,250)
    plt.colorbar()

def elim(a):
    for i in range(0, np.size(a)):
        for j in range(0, np.size(a)):
            b = np.size(a)
            if a[b-j-1] == 0 and (b-j-1) > 0:
                #print(np.size(a), (b-j-1))
                a.pop(np.size(a)-j-1)
                #print(a)
    return a

# while loop should be the outer most one
# naming the OD pairs
## storing the dictionary data to dcar variable


# gives the number of paths from the particular source and destination
od_routes = []
all_routes = []
tempalr = 0
for odp in range(0, len(p)):
    od_routes.append(np.size(p[odp]))
    tempalr = tempalr + np.size(p[odp])
    all_routes.append(tempalr)

q_car = np.zeros([np.size(pp), np.size(t)])
q_bike = np.zeros([np.size(pp), np.size(t)])
length=[0]*len(pp)
print(length)
err_car = np.zeros([np.size(t)])
err_bike = np.zeros([np.size(t)])
k_car = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
k_bike = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
v_car = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
v_bike = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
tt_car = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
tt_bike = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
dtrav_car = np.zeros((len(pp), len(DG.edges)+1, np.size(t)))
dtrav_bike = np.zeros((len(pp), len(DG.edges)+1, np.size(t)))
dtravp_bike = np.zeros([len(pp), np.size(t)])
dtravp_car = np.zeros([len(pp), np.size(t)])
lflow_car = np.zeros((np.size(t),len(DG.edges)+1,len(pp)))
lflow_bike = np.zeros((np.size(t),len(DG.edges)+1,len(pp)))
lf_car = np.zeros((len(DG.edges)+1,np.size(t)))
lf_bike = np.zeros((len(DG.edges)+1,np.size(t)))
kshrt_car = np.zeros((np.size(t),len(p) ))
kshrt_bike = np.zeros((np.size(t),len(p) ))
od_dmnd = 0
for i in range(0, np.size(d_car)):
    od_dmnd = od_dmnd + d_car[i]
print(od_dmnd)
## MSA algorithm application for the chosen O-D pair from the network
print(d_car)
print(lnks)
count=2
while True:
## path chosen from the possible paths avaliable from chosen O-D pair
# for odp in range(0, len(p)):
    if count==2:
        for ind in range(0, all_routes[0]):
            for m in range(0, np.size(t)):
                q_car[ind][m] = d_car[0]/np.size(p[0])
                q_bike[ind][m] = d_bike[0]/np.size(p[0]) 
        for i in range(1, len(p)):
            for ind in range(all_routes[i-1], all_routes[i]):
                for m in range(0, np.size(t)):
                    q_car[ind][m] = d_car[i]/np.size(p[i])
                    q_bike[ind][m] = d_bike[i]/np.size(p[i]) 

            ## link in the path chosen, density matrix calculation

    for path in pp:
        temp3 = 0
        prev=-1
        ind = pp.index(path)
        print(path, ind)
        for edge in path:
            data = DG.get_edge_data(edge[0],edge[1])
            # edge with start node and end node data
            num = data['label']
            temp3+= data['length']
            edglen = np.linspace( 0, data['length'], num = int((data['length'])/dx) )
            # stores length of the edge
            srt = edge[0]
            dst = edge[1]
            #print('data, num, temp3', data, num, temp3, prev, srt, dst)
            cal_densty(k_car, k_bike, num,srt,dst,ind,prev,edglen)

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
        for m in range(1,np.size(t)):
            length[ind] = temp3
    for ind in range(0, np.size(pp)):
        for nu in lnks:
            for i in range(1, np.size(t)+1):
                ddt_bike.append(d_trav_tim(x,tt_bike, i))
                ddt_car.append(d_trav_tim(x,tt_car, i))
            dtrav_bike[ind][nu][:] = ddt_bike
            dtrav_car[ind][nu][:] = ddt_car
            ddt_bike = []
            ddt_car = []
    print(np.size(dtrav_car))
    #print(dtrav_car[0][:][:])
    for path in pp:
        ind = pp.index(path)
        for i in range(0, np.size(t)):
            for edge in path:
                data = DG.get_edge_data(edge[0],edge[1])
                num = data['label']
                #print(p, ind, i, num)
                dtravp_bike[ind][i] = dtravp_bike[ind][i] + int(dtrav_bike[ind][num][i])
                dtravp_car[ind][i] = dtravp_car[ind][i] + int(dtrav_car[ind][num][i])

    temp = 0
    for ind in range(0, np.size(pp)):
        for i in range(0, np.size(t)):
            temp = temp + ( ( dtravp_car[ind][i] * q_car[ind][i] ) / 3600 )
        avg_trv_car.append((temp/100))
        temp = 0
    tempb = 0
    for ind in range(0, np.size(pp)):
        for i in range(0, np.size(t)):
            tempb = tempb + ( ( dtravp_bike[ind][i] * q_bike[ind][i] ) / 3600 )
        avg_trv_bike.append((tempb/100))
        tempb = 0    
    sys_trv_timb = 0
    sys_trv_timc = 0
    for i in range(0, np.size(avg_trv_bike)):
        sys_trv_timb = sys_trv_timb + avg_trv_bike[i]
        sys_trv_timc = sys_trv_timc + avg_trv_car[i]

    overal_sys_trv_timb = overal_sys_trv_timb + sys_trv_timb
    overal_sys_trv_timc = overal_sys_trv_timc + sys_trv_timc

    overal_sys_trv_tim = overal_sys_trv_timb + overal_sys_trv_timc
    if count > 1:
        print("overall_sys_trv_time: cars, bikes, total")
        print(overal_sys_trv_timc) 
        print(overal_sys_trv_timb)
        print(overal_sys_trv_tim)
        print(sys_trv_timc) 
        print(sys_trv_timb)
        print(sys_trv_timc+sys_trv_timb)


    # Spectral
    if count == 11 or count == 12:
        clrs = ["blue", "orange","green","red","purple","brown","pink","gray","olive","cyan","black", "yellow", "lime", "magenta", 
                "navajowhite","blue", "orange","green","red","purple","brown","pink","gray","olive","cyan","black", "yellow", "lime", 
                "magenta", "navajowhite","blue", "orange","green","red","purple","brown","pink","gray","olive","cyan","black", 
                "yellow", "lime", "magenta", "navajowhite","blue", "orange","green","red","purple","brown","pink","gray","olive",
                "cyan","black", "yellow", "lime", "magenta", "navajowhite","blue", "orange","green","red","purple","brown","pink",
                "gray","olive","cyan","black", "yellow", "lime", "magenta", "navajowhite","blue", "orange","green","red","purple",
                "brown","pink","gray","olive","cyan","black", "yellow", "lime", "magenta", "navajowhite","blue", "orange","green",
                "red","purple","brown","pink","gray","olive","cyan","black", "yellow", "lime", "magenta", "navajowhite","blue", 
                "orange","green","red","purple","brown","pink","gray","olive","cyan","black", "yellow", "lime", "magenta", "navajowhite"]
        fig = plt.figure(figsize=(8, 6), dpi=80)
        tx1 = fig.add_subplot(211)
        for i in range(0, np.size(pp)):
            plt.plot(range(0,np.size(dtravp_bike[i][:])),dtravp_bike[i][:],color=clrs[i], label = i)
        tx1.set(title = '', xlabel = 'departure time', ylabel = 'travel time of bikes (sec)')
        tx1.set_xlim([0, 75])
        tx1.set_ylim([0, 3500])
        tx1.legend(loc = 'center left', bbox_to_anchor=(1.001, 0.5))

        tx2 = fig.add_subplot(212)
        for i in range(0, np.size(pp)):
            plt.plot(range(0,np.size(dtravp_car[i][:])),dtravp_car[i][:],color=clrs[i], label = i)

        tx2.set_xlim([0, 75])
        tx2.set_ylim([0, 3500])
        tx2.set(title = '', xlabel = 'departure time', ylabel = 'travel time of cars (sec)')
        tx2.legend(loc = 'center left', bbox_to_anchor=(1.001, 0.5))
        plt.show()
        break
    for ts in range(0, np.size(t)):
        for rt in range(0, np.size(pp)):
            for l in lnks:
                lflow_bike[ts][l][rt] = q_bike[rt][ts]
                lflow_car[ts][l][rt] = q_car[rt][ts]
    for ts in range(0, np.size(t)):
        for l in lnks:
            temp_lfb = 0
            temp_lfc = 0
            for rt in range(0, np.size(pp)):
                temp_lfb = temp_lfb + lflow_bike[ts][l][rt]
                temp_lfc = temp_lfc + lflow_car[ts][l][rt]
            lf_bike[l][ts] = temp_lfb
            lf_car[l][ts] = temp_lfc

    temp1_bike = np.transpose(dtravp_bike)
    temp1_car = np.transpose(dtravp_car)

    for i in range(0, len(p)):
        for ind in range(all_routes[i-1], all_routes[i]):
            for m in range(0, np.size(t)):
                kshrt_car[m][i] = min(temp1_car[m][all_routes[i-1] : (all_routes[i]-1) ])
                kshrt_bike[m][i] = min(temp1_bike[m][all_routes[i-1] : (all_routes[i]-1) ])
    for ind in range(0, all_routes[0]):
        for m in range(0, np.size(t)):
            kshrt_car[m][0] = min(temp1_car[m][all_routes[i-1] : (all_routes[i]-1) ])
            kshrt_bike[m][0] = min(temp1_bike[m][all_routes[i-1] : (all_routes[i]-1) ])
    print(kshrt_bike)
    print(kshrt_car)
    print(lf_car[1][:])
    print(lflow_car[1][:][1])

    ## storing the error 
    for m in range(0, np.size(t)):
        nmrtr1_bike = 0
        nmrtr1_car = 0
        nmrtr2_bike = 0
        nmrtr2_car = 0
        for ind in range(0, len(p)):
            nmrtr2_bike = kshrt_bike[m][ind] * d_bike[ind] + nmrtr2_bike
            nmrtr2_car = kshrt_car[m][ind] * d_car[ind] + nmrtr2_car
        for l in lnks:
            nmrtr1_bike = (dtrav_bike[0][l][m] * lf_bike[l][m]) + nmrtr1_bike
            nmrtr1_car = (dtrav_car[0][l][m] * lf_car[l][m]) + nmrtr1_car


        err_car[m] = (nmrtr1_car-nmrtr2_car) / od_dmnd
        err_bike[m] = (nmrtr1_bike-nmrtr2_bike) / od_dmnd

    error_car = max(err_car)
    error_bike = max(err_bike)
    ercar.append(error_car)
    erbike.append(error_bike)
    ercarx = np.size(ercar)

    if count == 10 or count == 9 or (error_car < 0.002 and error_bike < 0.002):
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
    if count == 200 or count == 250:
        fig = plt.figure()
        tx1 = fig.add_subplot(221)
        plt.plot(range(0,np.size(dtravp_bike[0][:])),dtravp_bike[0][:],color="blue")
        plt.plot(range(0,np.size(dtravp_bike[1][:])),dtravp_bike[1][:],color="orange")
        plt.plot(range(0,np.size(dtravp_bike[2][:])),dtravp_bike[2][:],color="green")
        plt.plot(range(0,np.size(dtravp_bike[3][:])),dtravp_bike[3][:],color="red")
        plt.plot(range(0,np.size(dtravp_bike[4][:])),dtravp_bike[4][:],color="purple")
        plt.plot(range(0,np.size(dtravp_bike[5][:])),dtravp_bike[5][:],color="brown")
        plt.plot(range(0,np.size(dtravp_bike[6][:])),dtravp_bike[6][:],color="pink")
        plt.plot(range(0,np.size(dtravp_bike[7][:])),dtravp_bike[7][:],color="gray")
        plt.plot(range(0,np.size(dtravp_bike[8][:])),dtravp_bike[8][:],color="olive")
        plt.plot(range(0,np.size(dtravp_bike[9][:])),dtravp_bike[9][:],color="cyan")
        plt.plot(range(0,np.size(dtravp_bike[10][:])),dtravp_bike[10][:],color="black")
        tx1.set(title = 'travel time', xlabel = 'simulation time', ylabel = 'bike')
        tx1.set_xlim([0, 100])

        tx2 = fig.add_subplot(222)
        plt.plot(range(0,np.size(dtravp_car[0][:])),dtravp_car[0][:],color="blue", label = '0')
        plt.plot(range(0,np.size(dtravp_car[1][:])),dtravp_car[1][:],color="orange", label = '1')
        plt.plot(range(0,np.size(dtravp_car[2][:])),dtravp_car[2][:],color="green", label = '2')
        plt.plot(range(0,np.size(dtravp_car[3][:])),dtravp_car[3][:],color="red", label = '3')
        plt.plot(range(0,np.size(dtravp_car[4][:])),dtravp_car[4][:],color="purple", label = '4')
        plt.plot(range(0,np.size(dtravp_car[5][:])),dtravp_car[5][:],color="brown", label = '5')
        plt.plot(range(0,np.size(dtravp_car[6][:])),dtravp_car[6][:],color="pink", label = '6')
        plt.plot(range(0,np.size(dtravp_car[7][:])),dtravp_car[7][:],color="gray", label = '7')
        plt.plot(range(0,np.size(dtravp_car[8][:])),dtravp_car[8][:],color="olive", label = '8')
        plt.plot(range(0,np.size(dtravp_car[9][:])),dtravp_car[9][:],color="cyan", label = '9')
        plt.plot(range(0,np.size(dtravp_car[10][:])),dtravp_car[10][:],color="black", label = '10')
        tx2.set_xlim([0, 100])
        tx2.set(title = count, xlabel = 'simulation time', ylabel = 'car')
        plt.legend()
        plt.show()
    if (error_car < 0.002 and error_bike < 0.002):
        break

    ## redistribution of flow
    for m in range(0, np.size(t)):
        temp2_car = min(temp1_car[m][:(all_routes[0]-1)])
        temp3_car = np.where(temp1_car[m][:(all_routes[0]-1)] == temp2_car)
        temp4_car = temp3_car[0]
        min1 = temp4_car[0]
        q_car[min1][m] = q_car[min1][m]*((count-2)/(count-1))+(d_car[0]/(count-1))
        for i in range(0,all_routes[0]):
            #print("i",i)
            if(i!=min1):
                q_car[i][m] = q_car[i][m]*((count-2)/(count-1)) 

        temp2_bike = min(temp1_bike[m][:(all_routes[0]-1)])
        temp3_bike = np.where(temp1_bike[m][:(all_routes[0]-1)] == temp2_bike)
        temp4_bike = temp3_bike[0]
        min2 = temp4_bike[0]
        q_bike[min2][m] = q_bike[min2][m]*((count-2)/(count-1))+(d_bike[0]/(count-1))
        for i in range(0,all_routes[0]):
            if(i!=min2):
                q_bike[i][m] =q_bike[i][m]*((count-2)/(count-1))
    for k in range(1, len(p)):
        for m in range(0, np.size(t)):
            temp2_car = min(temp1_car[m][all_routes[k-1] : (all_routes[k]-1) ])
            temp3_car = np.where(temp1_car[m][all_routes[k-1] : (all_routes[k]-1) ] == temp2_car)
            temp4_car = temp3_car[0]
            min1 = temp4_car[0]     
            q_car[min1][m] = q_car[min1][m]*((count-2)/(count-1))+(d_car[k]/(count-1))
            for j in range(1, len(p)):
                for i in range(all_routes[j-1], all_routes[j]):
                    if(i!=min1):
                        q_car[i][m] = q_car[i][m]*((count-2)/(count-1)) 

            temp2_bike = min(temp1_bike[m][all_routes[k-1] : (all_routes[k]-1) ])
            temp3_bike = np.where(temp1_bike[m][all_routes[k-1] : (all_routes[k]-1) ] == temp2_bike)
            temp4_bike = temp3_bike[0]
            min2 = temp4_bike[0]
            q_bike[min2][m] = q_bike[min2][m]*((count-2)/(count-1))+(d_bike[k]/(count-1))
            for j in range(1, len(p)):
                for i in range(all_routes[j-1], all_routes[j]):
                    if(i!=min2):
                        q_bike[i][m] =q_bike[i][m]*((count-2)/(count-1))
## stopping criteria

    count+=1
    #print('t_bike',t_bike)

    length=[0]*len(pp)
    err_car = np.zeros([np.size(t)])
    err_bike = np.zeros([np.size(t)])
    k_car = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
    k_bike = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
    v_car = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
    v_bike = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
    tt_car = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
    tt_bike = np.zeros((len(pp),len(DG.edges)+1,np.size(x),np.size(t)))
    dtrav_car = np.zeros((len(pp), len(DG.edges)+1, np.size(t)))
    dtrav_bike = np.zeros((len(pp), len(DG.edges)+1, np.size(t)))
    dtravp_bike = np.zeros([len(pp), np.size(t)])
    dtravp_car = np.zeros([len(pp), np.size(t)])
    avg_trv = []
    avg_trv_car = []
    avg_trv_bike = []
    overal_sys_trv_timb = 0
    overal_sys_trv_timc = 0
    overal_sys_trv_tim = 0

    for i in range(0, np.size(x)):
        k_car[0][0][i][0] = 0.15
        k_bike[0][0][i][0] = 0.1875

    print('count',count)
    #print('q_bike', q_bike)

    if count == 1000:
        break

# travel time
# calibration 
# density 

time_elapsed = (time.perf_counter() - time_start)
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ( "------======-------" ,  "%5.1f secs %5.1f MByte" % (time_elapsed,memMb),  "------======-------" )
