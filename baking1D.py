########################
#MAIN PROGRAMME
#######################
from bakingFunctions import *
import numpy as np
import Image
from lparams import *
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pylab

theta=0;
dx=0.01/np.float32(N); 
dt=30;
Time=5400;
M=Time/np.float32(dt);
print M
print N
# N is number of spacial nodes
# M is number of temporal nodes
#**********************************
#% inputting the initial values
#**********************************

T = np.zeros((N+2)).astype(np.float32)
V = np.zeros((N+2)).astype(np.float32)
W = np.zeros((N+2)).astype(np.float32)

T1 = np.zeros((M+2,N+2)).astype(np.float32)
V1 = np.zeros((M+2,N+2)).astype(np.float32)
W1 = np.zeros((M+2,N+2)).astype(np.float32)

# transform from polar coordinates to x,y
def transform(W):
    global N
    print W[20]
    t = 20


    MM = N
    N = 356

    #bakk = W[t]
    arr = np.zeros((N+1,N+1)).astype(np.float32)

    # interpolation in order to keep baking in 32 array elements (for stability)
    bakk = interp1d(range(MM+1), W[t], kind='cubic')

    for i in range(-N/2,N/2+1):
        for j in range(-N/2,N/2+1):
            i2 = i
            j2 = j
            r = np.sqrt(i2*i2+j2*j2).astype(np.float32)
            if(r < N and r >= 0):
                arr[N/2-i,N/2-j] = bakk((N-r)*MM/N)

    if(False):
        I2 = Image.frombuffer('L',(N+1,N+1), (arr).astype(np.uint8),'raw','L',0,1)
        imgplot = plt.imshow(I2)
        plt.colorbar()
        gx, gy = np.gradient(arr)
        pylab.quiver(gx,gy)
        pylab.show()

        plt.show()
    #print "Saving Image res.png..."
    #I2.save('res.png')
    return arr

def calc():
    print "Baking Computation..."
    global T,V,W
    # initial conditions
    for i in range(0,N+1):
        T[i]=25
        V[i]=0
        W[i]=0.4061
        T1[0,i]=T[i]
        V1[0,i]=V[i]
        W1[0,i]=W[i]

    for t in range(0,np.int(M)+1):
        T_new=Tnew(T,V,W,N,dt,dx,theta)
        V_temp,W_temp,V_s,P=correction(T_new,V,W,N) #,P) ;
        V_new=Vnew(T_new,V_temp,W_temp,dx,dt,N,theta)
        V_new,W_temp=Correction2(T_new,V_new,W_temp,V_s,N,P,W)
        W_new=Wnew(T_new,V_new,W_temp,dx,dt,N,theta)
        T[0:N+1]=T_new[0:N+1]
        V[0:N+1]=V_new[0:N+1]
        W[0:N+1]=W_new[0:N+1]
        for i in range(0,N+1):
            T1[t+1,i]=T[i]
            V1[t+1,i]=V[i]
            W1[t+1,i]=W[i]

    Times = np.zeros((M+1,N+1)).astype(np.float32)
    T = np.zeros((M+1,N+1)).astype(np.float32)
    V = np.zeros((M+1,N+1)).astype(np.float32)
    W = np.zeros((M+1,N+1)).astype(np.float32)

    for t in range(0,np.int(M)+1):
        for i in range(0,N+1):
            l=(t)*dt/np.float32(60)
            x=(i)*dx
            Times[t,i]=l
            T[t,i]=T1[t,i]
            V[t,i]=V1[t,i]
            W[t,i]=W1[t,i]

    #print Times.shape
    #print T.shape
    #print Times
    #plt.plot(Times,T)
    #plt.show()

    #plt.plot(Times,V)
    #plt.show()

    #plt.plot(Times,W)
    #plt.show()

    # Transform W to 2D
    #newW = transform(W) # 2D
    return transform(T)

