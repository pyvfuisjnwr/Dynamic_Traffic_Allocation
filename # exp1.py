# exp1

# 2 Dynamic TT
# DTA implementation on 2 link network
# W W
# graphs
# density in first column
# organised
# contour graph
# velocity matrix and tt matrix from k matrix
# dyn trav tim 17-nov
# ww for heterogenous 
import time
import resource 
import numpy as np
from matplotlib import cbook
import matplotlib.pyplot as plt
import sys
import csv
np.set_printoptions(threshold=sys.maxsize)
from scipy.stats import gaussian_kde
time_start = time.perf_counter()

#########     INITIALISATION     #########    
dl = 200
beta_not_car = 0.94
beta_not_bike = 0.98
beta_car = 0.86
beta_bike = 0.52
vsf_bike = 80
vsf_car = 100

length = np.array([10,12])
dx_1 = 0.5
dx_2 = 0.5
dt = 0.005
num1 = int(((length[0]/dx_1)+1))
num2 = int(((length[1]/dx_2)+1))
num3 = int(1/dt)
print(num1, num2)
x1 = np.linspace(0,length[0],num= num1 ) 
x2 = np.linspace(0,length[1],num= num2 )
t = np.linspace(0,1,num = num3)
dts = dt * 3600
q_car = 12000
q_bike = 12000
k1_car= np.zeros([np.size(x1),np.size(t)])
k1_bike = np.zeros([np.size(x1),np.size(t)])
k2_car= np.zeros([np.size(x2),np.size(t)])
k2_bike = np.zeros([np.size(x2),np.size(t)])
inflo1_bike = np.zeros([np.size(x1),np.size(t)])
inflo1_car = np.zeros([np.size(x1),np.size(t)])
inflo2_bike = np.zeros([np.size(x2),np.size(t)])
inflo2_car = np.zeros([np.size(x2),np.size(t)])

qr1_car= np.zeros([np.size(x1),np.size(t)])
qr1_bike = np.zeros([np.size(x1),np.size(t)])
qr2_bike = np.zeros([np.size(x2),np.size(t)])
qr2_car = np.zeros([np.size(x2),np.size(t)])

qr1c = np.zeros([np.size(t)])

k2 = np.zeros([np.size(x2),np.size(t)])
k1 = np.zeros([np.size(x1),np.size(t)])

q1_car = np.zeros([np.size(t)])
q1_bike = np.zeros([np.size(t)])
q2_car = np.zeros([np.size(t)])
q2_bike = np.zeros([np.size(t)])

lmnoc_r1 = []
lmnoc_r2 = []
lmnob_r1 = []
lmnob_r2 = []

# for all space steps at first time step
# initial condition

a = beta_not_bike
b = beta_not_car
c = beta_bike
d = beta_car
# k is 187.5 for homogenous case, obtained by solving

for m in range(np.size(t)):
    q1_car[m] = q_car/2
    q1_bike[m] = q_bike/2
    q2_car[m] = q_car/2
    q2_bike[m] = q_bike/2

v1_car = np.zeros([np.size(x1),np.size(t)])
v1_bike = np.zeros([np.size(x1),np.size(t)])
v2_car = np.zeros([np.size(x2),np.size(t)])
v2_bike = np.zeros([np.size(x2),np.size(t)])

tt1_car = np.zeros([np.size(x1),np.size(t)])
tt1_bike = np.zeros([np.size(x1),np.size(t)])
tt2_car = np.zeros([np.size(x2),np.size(t)])
tt2_bike = np.zeros([np.size(x2),np.size(t)])

dt1_car = []
dt1_bike = []
dt2_car = []
dt2_bike = []

ddt1_car = []
ddt1_bike = []
ddt2_car = []
ddt2_bike = []

ercar = []
erbike = []

err_bike = np.zeros([np.size(t)])
err_car = np.zeros([np.size(t)])
k1_bike[np.size(x1)-2+1][0] = 0
k1_car[np.size(x1)-2+1][0] = 0
k2_bike[np.size(x2)-2+1][0] = 0
k2_car[np.size(x2)-2+1][0] = 0
#print(k1_car)
# print(np.size(t), np.size(x1))
def writecsvkcar(a):
    with open("k car.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(a)
def writecsvvelcar(a):
    with open("vel car.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(a)
def writecsvttcar(a):
    with open("tt car.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(a)
######################################################################################################
count = 2
while True:
    for m in range(1, np.size(t)):
        k1_bike[0][m] = q1_bike[m] / vsf_bike
        k1_car[0][m] = q1_car[m] / vsf_car
        k2_bike[0][m] = q2_bike[m] / vsf_bike
        k2_car[0][m] = q2_car[m] / vsf_car
        #k2_car[0][m] = ( 156800 + ( ( 156800*156800 - 4*832*q2_car[m]*10 ) ** (1/2) ) ) / (2*832)
        #print('flow', m, k1_bike[0][m], q1_bike[m] , k1_car[0][m], q1_car[m]) 
        #print('flow', m, k2_bike[0][m], q2_bike[m], k2_car[0][m], q2_bike[m])

        # the above quadratic eqn is used for homogenous case alone, k calculation
        # first row initialisation

#########     DENSITY CALCULATION USING THE ATD MODEL     #########  
    for m in range(1, np.size(t)):
  
        for n in range(1,(np.size(x1)-1)):
            e1 = k1_bike[n-1][m-1]
            #f1 = k1_bike[n-1][m]
            g1 = k1_bike[n+1][m-1]
            #print("======+++=====" , e1)

            h1 = k1_car[n-1][m-1]
            #u1 = k1_car[n-1][m]
            v1 = k1_car[n+1][m-1]

            r1 = g1+v1
            s1 = e1+h1

            funct_1 = min((vsf_bike/dl)*(e1*(a*dl-c*s1)) , 3476)
            if funct_1 < 0:
                funct_1=0
            funct_2 = min((vsf_bike/dl)*(g1*(a*dl-c*r1)), 3476)
            if funct_2 < 0:
                funct_2=0
            funct_3 = min((vsf_car/dl)*(h1*(b*dl-d*s1)),1942)
            if funct_3 < 0:
                funct_3=0            
            funct_4 = min((vsf_car/dl)*(v1*(b*dl-d*r1)),1942)
            if funct_4 < 0:
                funct_4=0

            k1_bike[n][m] = (0.5*(e1+g1))-((0.5*dt/dx_1)*(funct_2-funct_1))
            k1_car[n][m] = (0.5*(h1+v1))-((0.5*dt/dx_1)*(funct_4-funct_3))
        
            k1_bike[np.size(x1)-2+1][m] = k1_bike[np.size(x1)-2][m]
            k1_car[np.size(x1)-2+1][m] = k1_car[np.size(x1)-2][m]

        for o in range(1,(np.size(x2)-1)):
            e2 = k2_bike[o-1][m-1]
            f2 = k2_bike[o-1][m]
            g2 = k2_bike[o+1][m-1]
            h2 = k2_car[o-1][m-1]
            u2 = k2_car[o-1][m]
            v2 = k2_car[o+1][m-1]
            p2 = f2+u2
            r2 = g2+v2
            s2 = e2+h2

            ffunct_1 = min((vsf_bike/dl)*(e2*(a*dl-c*s2)), 3476)
            if ffunct_1 < 0:
                ffunct_1=0            
            ffunct_2 = min((vsf_bike/dl)*(g2*(a*dl-c*r2)), 3476)
            if ffunct_2 < 0:
                ffunct_2=0
            ffunct_3 = min((vsf_car/dl)*(h2*(b*dl-d*s2)),1942)
            if ffunct_3 < 0:
                ffunct_3=0            
            ffunct_4 = min((vsf_car/dl)*(v2*(b*dl-d*r2)),1942)
            if ffunct_4 < 0:
                ffunct_4=0
            k2_bike[o][m] = (0.5*(e2+g2))-((0.5*dt/dx_2)*(ffunct_2-ffunct_1))
            k2_car[o][m] = (0.5*(h2+v2))-((0.5*dt/dx_2)*(ffunct_4-ffunct_3))

            k2_bike[np.size(x2)-2+1][m] = k2_bike[np.size(x2)-2][m]
            k2_car[np.size(x2)-2+1][m] = k2_car[np.size(x2)-2][m]

    #writecsvflowcar(k1_bike)  
       
        #if count == 3:
            #print(count, m, k1_car)
#########    VELOCITY and  TRAVEL TIME CALCULATION     #########  

    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x1))):
            # velocity matrix
            if k1_car[p][m] < 0.1:
                v1_car[p][m] = vsf_car
            else:
                #print(v1_car[p][m], q1_car[m], k1_car[p][m])
                v1_car[p][m] = vsf_car * ( b * dl  - d * ( k1_car[p][m] + k1_bike[p][m] ) ) / dl
            if k1_bike[p][m] < 0.1:
                v1_bike[p][m] = vsf_bike
            else:
                v1_bike[p][m] = vsf_bike * ( a * dl  - c * ( k1_car[p][m] + k1_bike[p][m] ) ) / dl
            if v1_car[p][m] > 100:
                v1_car[p][m] = vsf_car
            if v1_bike[p][m] > 80:
                v1_bike[p][m] = vsf_bike
            if v1_car[p][m] < 0.1:
                v1_car[p][m] = 0.1
            if v1_bike[p][m] < 0.1:
                v1_bike[p][m] = 0.1
            #print(v1_car[p][m], q1_car[m], k1_car[p][m])

            tt1_car[p][m] = (dx_1)*3600/v1_car[p][m]
            tt1_bike[p][m] = (dx_1)*3600/v1_bike[p][m]

        for q in range(0, (np.size(x2))):
            if k2_car[q][m] < 0.1:
                v2_car[q][m] = vsf_car
            else:    
                v2_car[q][m] = vsf_car * ( b * dl  - d * ( k2_car[p][m] + k2_bike[p][m] ) ) / dl
            if k2_bike[q][m] < 0.1:
                v2_bike[q][m] = vsf_bike
            else:
                v2_bike[q][m] = vsf_bike * ( a * dl  - c * ( k2_car[p][m] + k2_bike[p][m] ) ) / dl

            if v2_car[q][m] > 100:
                v2_car[q][m] = vsf_car
            if v2_bike[q][m] > 80:
                v2_bike[q][m] = vsf_bike
            if v2_car[q][m] < 0.1:
                v2_car[q][m] = 0.1
            if v2_bike[q][m] < 0.1:
                v2_bike[q][m] = 0.1
            # travel time matrix 
            tt2_car[q][m] = (dx_2)*3600/v2_car[q][m]
            tt2_bike[q][m] = (dx_2)*3600/v2_bike[q][m]
    # writecsvvelcar(v2_car)

    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x1))):
                inflo1_bike[p][m] = k1_bike[p][m] * v1_bike[p][m]
                inflo1_car[p][m] = k1_car[p][m] * v1_car[p][m]

    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x2))):
                inflo2_bike[p][m] = k2_bike[p][m] * v2_bike[p][m]
                inflo2_car[p][m] = k2_car[p][m] * v2_car[p][m]
    
    #print("inflo1_bike, inflo2_bike", inflo1_bike, inflo2_bike)
    writecsvkcar(inflo2_car)

        # plt.plot(range(0,m),t1_car,'o-')
    #print(v1_car)

#########     DYNAMIC TRAVEL TIME CALCULATION     #########    
# n - first row of instantaneous travel time matrix
# a is the matrix (inst tt)    

    def trav_tim(x1,a,col):
        trv_tim = []
        tot_tim = 0
        for i in range(0, np.size(x1)):
            for j in range(0, 1):
                if col >= np.size(t):
                    break
                tt = int(a[i][col-1])
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
                tt = int(a[i][col-1])
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

    for i in range(1, np.size(t)+1):
        dt1_bike.append(trav_tim(x1,tt1_bike, i))
    for i in range(1, np.size(t)+1):
        dt1_car.append(trav_tim(x1,tt1_car, i))
    #print("dt1_car",dt1_car)
    for i in range(1, np.size(t)+1):
        dt2_bike.append(trav_tim(x2,tt2_bike, i))
        #print(i, dt2_bike)
    for i in range(1, np.size(t)+1):
        dt2_car.append(trav_tim(x2,tt2_car, i))
    #writecsvttcar(dt1_bike)
    for i in range(1, np.size(t)+1):
        ddt1_bike.append(d_trav_tim(x1,tt1_bike, i))
    for i in range(1, np.size(t)+1):
        ddt1_car.append(d_trav_tim(x1,tt1_car, i))
    for i in range(1, np.size(t)+1):
        ddt2_bike.append(d_trav_tim(x2,tt2_bike, i))
    for i in range(1, np.size(t)+1):
        ddt2_car.append(d_trav_tim(x2,tt2_car, i))

    a1 = dt1_car[:]
    a2 = dt1_bike[:]
    a3 = dt2_car[:]
    a4 = dt2_bike[:]
    #print('ddt1_car', ddt1_bike, np.size(ddt1_bike))
    tc = np.zeros([len(dt1_car)])
    tb = np.zeros([len(dt1_bike)])
    # ddt1 comprises of all set of values
    # dt1 makes 0 for which cant reach destination 
    # elim function eliminates those which cant reach the destination
    #print('dt2_car', ddt2_bike)

    for i in range(0,len(dt1_car)):
        tc[i] = abs ( ( (a1[i]-a3[i]) / max(a1[i], a3[i]) ) * 100 )
    for i in range(0,len(dt1_bike)):
        tb[i] = abs ( ( (a2[i]-a4[i]) / max(a2[i], a4[i]) ) * 100 )

    elim(a1)
    elim(a2)
    elim(a3)
    elim(a4)
    #########    PLOTTING     #########    
    print('count', count)

    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x1))):
            k1[p][m] = k1_car[p][m] + k1_bike[p][m]
        for p in range(0, (np.size(x2))):
            k2[p][m] = k2_car[p][m] + k2_bike[p][m]     
    #project results
    for m in range(0, np.size(t)):    
        for p in range(0, (np.size(x1))):
            qr1_car[p][m] = k1_car[p][m] * v1_car[p][m]
            qr1_bike[p][m] = k1_bike[p][m] * v1_bike[p][m]
        for p in range(0, (np.size(x2))):
            qr2_car[p][m] = k2_car[p][m] * v2_car[p][m]
            qr2_bike[p][m] = k2_bike[p][m] * v2_bike[p][m]


    if count == 305:
        fig = plt.figure(figsize=(8, 6), dpi=80)
        t = np.linspace(0,1,num = num3)
        s = np.linspace(0,10,num= num1)
        r = k1_car[:,:]
        # evaluate on a regular grid
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = r
        ax1 = fig.add_subplot(111)
        # Plot the result as an image
        ax1.set_xlabel('Departure time (1 unit = 18 sec)')
        ax1.set_ylabel('Road stretch (km)')
        plt.imshow(Z.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 10],
                   cmap='OrRd')
        plt.clim(0,300)
        cb = plt.colorbar()
        cb.set_label('Density (veh/km)')
        plt.show()
        # Spectral
        fig = plt.figure(figsize=(8, 6), dpi=80)
        r2 = k2_car[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        s2 = np.linspace(0,12,num= num2)
        ax2 = fig.add_subplot(111)
        ax2.set_title('')
        ax2.set_xlabel('Departure time (1 unit = 18 sec)')
        ax2.set_ylabel('Road stretch (km)')
        plt.imshow(r2.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 12],
                   cmap='OrRd')
        plt.clim(0,300)
        cbar = plt.colorbar()
        cbar.set_label('Density (veh/km)')
        plt.show()

        fig = plt.figure(figsize=(8, 6), dpi=80)
        t = np.linspace(0,1,num = num3)
        s = np.linspace(0,10,num= num1)
        r = k1_bike[:,:]
        # evaluate on a regular grid
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = r
        ax1 = fig.add_subplot(111)
        # Plot the result as an image
        ax1.set_xlabel('Departure time (1 unit = 18 sec)')
        ax1.set_ylabel('Road stretch (km)')
        plt.imshow(Z.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 10],
                   cmap='OrRd')
        plt.clim(0,300)
        cb = plt.colorbar()
        cb.set_label('Density (veh/km)')
        plt.show()
        # Spectral

        fig = plt.figure(figsize=(8, 6), dpi=80)
        r2 = k2_bike[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        s2 = np.linspace(0,12,num= num2)
        ax2 = fig.add_subplot(111)
        ax2.set_title('')
        ax2.set_xlabel('Departure time (1 unit = 18 sec)')
        ax2.set_ylabel('Road stretch (km)')
        plt.imshow(r2.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 12],
                   cmap='OrRd')
        plt.clim(0,300)
        cbar = plt.colorbar()
        cbar.set_label('Density (veh/km)')
        plt.show()

        fig = plt.figure(figsize=(8, 6), dpi=80)
        t = np.linspace(0,1,num = num3)
        s = np.linspace(0,10,num= num1)
        r = k1[:,:]
        # evaluate on a regular grid
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = r
        ax1 = fig.add_subplot(111)
        # Plot the result as an image
        ax1.set_xlabel('Departure time (1 unit = 18 sec)')
        ax1.set_ylabel('Road stretch (km)')
        plt.imshow(Z.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 10],
                   cmap='OrRd')
        plt.clim(0,300)
        cb = plt.colorbar()
        cb.set_label('Density (veh/km)')
        plt.show()
        # Spectral

        fig = plt.figure(figsize=(8, 6), dpi=80)
        r2 = k2[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        s2 = np.linspace(0,12,num= num2)
        ax2 = fig.add_subplot(111)
        ax2.set_title('')
        ax2.set_xlabel('Departure time (1 unit = 18 sec)')
        ax2.set_ylabel('Road stretch (km)')
        plt.imshow(r2.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 12],
                   cmap='OrRd')
        plt.clim(0,300)
        cbar = plt.colorbar()
        cbar.set_label('Density (veh/km)')
        plt.show()

        fig = plt.figure(figsize=(8, 6), dpi=80)
        t = np.linspace(0,1,num = num3)
        s = np.linspace(0,10,num= num1)
        r = v1_car[:,:]
        # evaluate on a regular grid
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = r
        ax1 = fig.add_subplot(111)
        # Plot the result as an image
        ax1.set_xlabel('Departure time (1 unit = 18 sec)')
        ax1.set_ylabel('Road stretch (km)')
        plt.imshow(Z.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 10],
                   cmap='jet')
        plt.clim(0,100)
        cb = plt.colorbar()
        cb.set_label('Velocity (km/hr)')
        plt.show()
        # Spectral

        fig = plt.figure(figsize=(8, 6), dpi=80)
        r2 = v2_car[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        s2 = np.linspace(0,12,num= num2)
        ax2 = fig.add_subplot(111)
        ax2.set_title('')
        ax2.set_xlabel('Departure time (1 unit = 18 sec)')
        ax2.set_ylabel('Road stretch (km)')
        plt.imshow(r2.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 12],
                   cmap='jet')
        plt.clim(0,100)
        cbar = plt.colorbar()
        cbar.set_label('Velocity (km/hr)')
        plt.show()

        fig = plt.figure(figsize=(8, 6), dpi=80)
        t = np.linspace(0,1,num = num3)
        s = np.linspace(0,10,num= num1)
        r = v1_bike[:,:]
        # evaluate on a regular grid
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = r
        ax1 = fig.add_subplot(111)
        # Plot the result as an image
        ax1.set_xlabel('Departure time (1 unit = 18 sec)')
        ax1.set_ylabel('Road stretch (km)')
        plt.imshow(Z.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 10],
                   cmap='jet')
        plt.clim(0,100)
        cb = plt.colorbar()
        cb.set_label('Velocity (km/hr)')
        plt.show()
        # Spectral

        fig = plt.figure(figsize=(8, 6), dpi=80)
        r2 = v2_bike[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        s2 = np.linspace(0,12,num= num2)
        ax2 = fig.add_subplot(111)
        ax2.set_title('')
        ax2.set_xlabel('Departure time (1 unit = 18 sec)')
        ax2.set_ylabel('Road stretch (km)')
        plt.imshow(r2.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[0, 200, 0, 12],
                   cmap='jet')
        plt.clim(0,100)
        cbar = plt.colorbar()
        cbar.set_label('Velocity (km/hr)')
        plt.show()

    if count == 750:
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(28, 10), dpi=80)
        axes.flat[0].set_xlabel('',fontsize = 32)
        axes.flat[0].set_ylabel('Road stretch (km)',fontsize = 32)
        axes.flat[0].tick_params(axis='both', which='major', labelsize=32)
        axes.flat[0].set_title('Route 1',fontsize=32)
        s = np.linspace(0,10,num= num1)
        rt1 = k1_car[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt1
        im = axes.flat[0].imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',extent=[0, 200, 0, 10],cmap='OrRd',vmin=0, vmax=300)
        
        s2 = np.linspace(0,12,num= num2)
        rt2 = k2_car[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt2
        plt.xlabel('',fontsize = 32)
        plt.ylabel('',fontsize = 32)
        axes.flat[1].set_title('Route 2',fontsize=32)
        axes.flat[1].tick_params(axis='both', which='major', labelsize=32)
        im = axes.flat[1].imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto',extent=[0, 200, 0, 12], cmap='OrRd', vmin=0, vmax=300)
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation = "horizontal",aspect = 30).set_label('Departure time (1 unit = 18 sec)',fontsize = 32,labelpad=-120)
        im.figure.axes[2].tick_params(axis="x", labelsize=32)
        plt.show()             
    if count == 750:
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(28, 10), dpi=80)
        axes.flat[0].set_xlabel('',fontsize = 32)
        axes.flat[0].set_ylabel('Road stretch (km)',fontsize = 32)
        axes.flat[0].tick_params(axis='both', which='major', labelsize=32)
        axes.flat[0].set_title('Route 1',fontsize=32)
        s = np.linspace(0,10,num= num1)
        rt1 = k1_bike[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt1
        im = axes.flat[0].imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',extent=[0, 200, 0, 10],cmap='OrRd',vmin=0, vmax=300)
        
        s2 = np.linspace(0,12,num= num2)
        rt2 = k2_bike[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt2
        plt.xlabel('',fontsize = 32)
        plt.ylabel('',fontsize = 32)
        axes.flat[1].set_title('Route 2',fontsize=32)
        axes.flat[1].tick_params(axis='both', which='major', labelsize=32)
        im = axes.flat[1].imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto',extent=[0, 200, 0, 12], cmap='OrRd', vmin=0, vmax=300)
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation = "horizontal",aspect = 30).set_label('Departure time (1 unit = 18 sec)',fontsize = 32,labelpad=-120)
        im.figure.axes[2].tick_params(axis="x", labelsize=32)
        plt.show() 
    # velocity            
    if count == 750:
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(28, 10), dpi=80)
        axes.flat[0].set_xlabel('',fontsize = 32)
        axes.flat[0].set_ylabel('Road stretch (km)',fontsize = 32)
        axes.flat[0].tick_params(axis='both', which='major', labelsize=32)
        axes.flat[0].set_title('Route 1',fontsize=32)
        s = np.linspace(0,10,num= num1)
        rt1 = v1_car[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt1
        im = axes.flat[0].imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',extent=[0, 200, 0, 10],cmap='jet',vmin=0, vmax=100)
        
        s2 = np.linspace(0,12,num= num2)
        rt2 = v2_car[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt2
        plt.xlabel('',fontsize = 32)
        plt.ylabel('',fontsize = 32)
        axes.flat[1].set_title('Route 2',fontsize=32)
        axes.flat[1].tick_params(axis='both', which='major', labelsize=32)
        im = axes.flat[1].imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto',extent=[0, 200, 0, 12], cmap='jet', vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation = "horizontal",aspect = 30).set_label('Departure time (1 unit = 18 sec)',fontsize = 32,labelpad=-120)
        im.figure.axes[2].tick_params(axis="x", labelsize=32)
        plt.show() 
    if count == 750:
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(28, 10), dpi=80)
        axes.flat[0].set_xlabel('',fontsize = 32)
        axes.flat[0].set_ylabel('Road stretch (km)',fontsize = 32)
        axes.flat[0].tick_params(axis='both', which='major', labelsize=32)
        axes.flat[0].set_title('Route 1',fontsize=32)
        s = np.linspace(0,10,num= num1)
        rt1 = v1_bike[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt1
        im = axes.flat[0].imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',extent=[0, 200, 0, 10],cmap='jet',vmin=0, vmax=100)
        
        s2 = np.linspace(0,12,num= num2)
        rt2 = v2_bike[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt2
        plt.xlabel('',fontsize = 32)
        plt.ylabel('',fontsize = 32)
        axes.flat[1].set_title('Route 2',fontsize=32)
        axes.flat[1].tick_params(axis='both', which='major', labelsize=32)
        im = axes.flat[1].imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto',extent=[0, 200, 0, 12], cmap='jet', vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation = "horizontal",aspect = 30).set_label('Departure time (1 unit = 18 sec)',fontsize = 32,labelpad=-120)
        im.figure.axes[2].tick_params(axis="x", labelsize=32)
        plt.show() 
    # total density
    if count == 750:
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(28, 10), dpi=80)
        axes.flat[0].set_xlabel('',fontsize = 32)
        axes.flat[0].set_ylabel('Road stretch (km)',fontsize = 32)
        axes.flat[0].tick_params(axis='both', which='major', labelsize=32)
        axes.flat[0].set_title('Route 1',fontsize=32)
        s = np.linspace(0,10,num= num1)
        rt1 = k1[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 10, 21)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt1
        im = axes.flat[0].imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',extent=[0, 200, 0, 10],cmap='OrRd',vmin=0, vmax=300)
        
        s2 = np.linspace(0,12,num= num2)
        rt2 = k2[:,:]
        xgrid = np.linspace(0, 60, 200)
        ygrid = np.linspace(0, 12, 25)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = rt2
        plt.xlabel('',fontsize = 32)
        plt.ylabel('',fontsize = 32)
        axes.flat[1].set_title('Route 2',fontsize=32)
        axes.flat[1].tick_params(axis='both', which='major', labelsize=32)
        im = axes.flat[1].imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto',extent=[0, 200, 0, 12], cmap='OrRd', vmin=0, vmax=300)
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation = "horizontal",aspect = 30).set_label('Departure time (1 unit = 18 sec)',fontsize = 32,labelpad=-120)
        im.figure.axes[2].tick_params(axis="x", labelsize=32)
        plt.show() 
        
    # travel time
        fig = plt.figure(figsize=(28, 15), dpi=80)
        t1 = a1[:]
        t3 = a3[:]
        tx1 = fig.add_subplot(211)
        plt.plot(range(0,np.size(a1)),t1,color="black",linewidth=3, label = 'Route 1')
        plt.plot(range(0,np.size(a3)),t3,color="black",linestyle = 'dashed', linewidth=3, label = 'Route 2')
        plt.xticks(fontsize = 28)
        plt.yticks(fontsize = 28)
        tx1.set_ylim([0, 1600])
        tx1.set_xlim([0, 150])
        plt.xlabel('Departure time (1 unit = 18 sec)',fontsize=28)
        plt.ylabel('Travel Time of cars (sec)',fontsize=28)
        plt.legend(fontsize=28)

        t2 = a2[:]
        t4 = a4[:]
        tx2 = fig.add_subplot(212)
        plt.plot(range(0,np.size(a2)),t2,color="black", linewidth=3, label = 'Route 1')
        plt.plot(range(0,np.size(a4)),t4,color="black", linewidth=3,linestyle = 'dashed', label = 'Route 2')
        plt.xticks(fontsize = 28)
        plt.yticks(fontsize = 28)
        tx2.set_ylim([0, 1600])
        tx2.set_xlim([0, 150])
        plt.xlabel('Departure time (1 unit = 18 sec)',fontsize=28)
        plt.ylabel('Travel Time of TWs (sec)',fontsize=28)
        plt.legend(fontsize=28)
        plt.show() 
        break
        
    # relative gap
    if count == 4:
        fig = plt.figure(figsize=(28, 10), dpi=80)
        erc = ercar[:]
        erb = erbike[:]
        qx1 = fig.add_subplot(111)
        plt.plot(range(0,np.size(erc)),erc,color="black",label = 'Cars')
        plt.plot(range(0,np.size(erb)),erb,color="black",linestyle = 'dashed',  label = 'TWs')
        qx1.set_xlim([0, 80])
        plt.xticks(fontsize = 32)
        plt.yticks(fontsize = 32)
        plt.xlabel('Iterations',fontsize=32)
        plt.ylabel('Average excessive cost (AEC)',fontsize=32)
        plt.legend(fontsize=28)
        plt.show() 
        break

#########    VELOCITY     #########     


#########    DENSITY [w.r.t space steps]     #########     


#########    DEMAND IN EACH ROUTE [FLOWS]     #########     


    for m in range(0, np.size(t)):
        err_car[m] = ((ddt1_car[m]*q1_car[m])+(ddt2_car[m]*q2_car[m])- (min(ddt1_car[m],ddt2_car[m])*q_car))/(min(ddt1_car[m],ddt2_car[m])*q_car)
        err_bike[m] = ((ddt1_bike[m]*q1_bike[m])+(ddt2_bike[m]*q2_bike[m])- (min(ddt1_bike[m],ddt2_bike[m])*q_bike))/(min(ddt1_bike[m],ddt2_bike[m])*q_bike)

##########################              REDISTRIBUTING THE FLOW           #############################
    for m in range(0, np.size(t)):
        if(ddt1_car[m]<ddt2_car[m]):
            q1_car[m] = q1_car[m]*((count-2)/(count-1))+(q_car/(count-1))
            q2_car[m] = q2_car[m]*((count-2)/(count-1))
        else:
            q2_car[m] = q2_car[m]*((count-2)/(count-1))+(q_car/(count-1))
            q1_car[m] = q1_car[m]*((count-2)/(count-1))

        if(ddt1_bike[m]<ddt2_bike[m]):
            q1_bike[m] = q1_bike[m]*((count-2)/(count-1))+(q_bike/(count-1))
            q2_bike[m] = q2_bike[m]*((count-2)/(count-1))
        else:
            q2_bike[m] = q2_bike[m]*((count-2)/(count-1))+(q_car/(count-1))
            q1_bike[m] = q1_bike[m]*((count-2)/(count-1))
    #print(q1_car)
    #print(q2_car)


##########################                       ERROR                    #############################
    for m in range(0, np.size(t)):
        err_car[m] = ((ddt1_car[m]*q1_car[m])+(ddt2_car[m]*q2_car[m]) - (min(ddt1_car[m],ddt2_car[m])*q_car))/(min(ddt1_car[m],ddt2_car[m])*q_car)
        err_bike[m] = ((ddt1_bike[m]*q1_bike[m])+(ddt2_bike[m]*q2_bike[m]) - (min(ddt1_bike[m],ddt2_bike[m])*q_bike))/(min(ddt1_bike[m],ddt2_bike[m])*q_bike)
        #if m == 199 and count == 1:
    #print(err_car)

    #print(q1_car)
    #print(q2_car)

    error_car = max(err_car)
    lsd_c = list(err_car).index((error_car))
    error_bike = max(err_bike)
    lsd_b = list(err_bike).index((error_bike))
    ercar.append(error_car)
    erbike.append(error_bike)
    ercarx = np.size(ercar)
    lmnoc_r1.append(q1_car[lsd_c])
    lmnoc_r2.append(q2_car[lsd_c])
    lmnob_r1.append(q1_bike[lsd_b])
    lmnob_r2.append(q2_bike[lsd_b])
    #print(lmnoc_r1)
    #print(lmnob_r1)
    #print(lmnoc_r2)
    #print(lmnob_r2)

    if count == 200:
        fig = plt.figure(figsize=(28, 10), dpi=80)
        erc = lmnoc_r1[:]
        erb = lmnob_r1[:]
        qx1 = fig.add_subplot(111)
        plt.plot(range(0,np.size(erc)),erc,color="black",label = 'Cars')
        plt.plot(range(0,np.size(erb)),erb,color="black",linestyle = 'dashed',  label = 'Bikes')
        qx1.set_xlim([1, 200])
        plt.xticks(fontsize = 32)
        plt.yticks(fontsize = 32)
        plt.xlabel('Iterations',fontsize=32)
        plt.ylabel('Flows',fontsize=32)
        plt.legend(fontsize=28)
        plt.show() 
    if count == 200:
        fig = plt.figure(figsize=(28, 10), dpi=80)
        erc = lmnoc_r2[:]
        erb = lmnob_r2[:]
        qx1 = fig.add_subplot(111)
        plt.plot(range(0,np.size(erc)),erc,color="black",label = 'Cars')
        plt.plot(range(0,np.size(erb)),erb,color="black",linestyle = 'dashed',  label = 'Bikes')
        qx1.set_xlim([1, 200])
        plt.xticks(fontsize = 32)
        plt.yticks(fontsize = 32)
        plt.xlabel('Iterations',fontsize=32)
        plt.ylabel('Flows',fontsize=32)
        plt.legend(fontsize=28)
        plt.show() 
        break
    # print('count',count)
    #print('t1_car',t1_car)
    #print('t2_car',t2_car)
    #print('q1_car',q1_car)
    #print('q2_car',q2_car)
    #print(q1_bike)
    #print(err_car)
    #print('t1', t1_car)
    #print('t2', t2_car)
    #print(error_bike)
    #fig = plt.figure()
    #graph = fig.add_subplot(111)
##########################            STOPPING CRITERIA         #############################

    count+=1

##########################       SPACE STEPS [FIRST COLUMN]       ##########################

    k1_car= np.zeros([np.size(x1),np.size(t)])
    k1_bike = np.zeros([np.size(x1),np.size(t)])
    k2_car= np.zeros([np.size(x2),np.size(t)])
    k2_bike = np.zeros([np.size(x2),np.size(t)]) 



##########################         SETTING ARRAYS TO ZERO         ##########################

    v1_car = np.zeros([np.size(x1),np.size(t)])
    v1_bike = np.zeros([np.size(x1),np.size(t)])
    v2_car = np.zeros([np.size(x2),np.size(t)])
    v2_bike = np.zeros([np.size(x2),np.size(t)])

    dt1_car = []
    dt1_bike = []
    dt2_car = []
    dt2_bike = []
    ddt1_car = []
    ddt1_bike = []
    ddt2_car = []
    ddt2_bike = []
            
    err_bike = np.zeros([np.size(t)])
    err_car = np.zeros([np.size(t)])
    #print(np.round(t2_bike,3), np.round(t1_bike,3))
    #print('car ', np.round(t2_car,3), np.round(t1_car,3))
time_elapsed = (time.perf_counter() - time_start)
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ( "------======-------" ,  "%5.1f secs %5.1f MByte" % (time_elapsed,memMb),  "------======-------" )
