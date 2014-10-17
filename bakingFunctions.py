######################################
# Function to calculate New Temperature.
######################################
import numpy as np
from math import atan
from scipy import interpolate

def Tnew(T,V,W,N,dt,dx,theta):
#**********************************
#constants below
#**********************************
    k=0.07
    cp=3500
    sig=5.670*10**(-8)
    Dw=1.35*10**(-10)
    T_air=210
    T_r=210
    esp_p=0.9
    esp_r=0.9
    lam=2.261*10**(6)
    W_air=0
    hc=0.5

    #*********************************
    #loop starts
    #********************************
    a=np.zeros((N+1,N+1))
    b=np.zeros((N+1))
    for i in range(1,N):#i=2:N
        r=k*dt/((170+284*W[i])*cp*dx*dx)
        a[i,i-1]=-r*(1-theta)
        a[i,i]=1+2*r*(1-theta)
        a[i,i+1]=-r*(1-theta)
        b[i]=r*theta*T[i-1]+(1-2*r*theta)*T[i]+r*theta*T[i+1]+lam*Dw*dt/(cp*dx*dx)*(W[i+1]-2*W[i]+W[i-1])
    #**********************************************
    #for temp at 1st node where T_f is fictious node
    #**********************************************
    a1=(12/np.float32(5.6))
    b1= (12/np.float32(5.6))
    a2=1+a1*a1
    b2=1+b1*b1
    F_sp=(2./(np.pi*a1*b1))*(np.log(np.sqrt(a2*b2/(1+a1*a1+b1*b1)))+a1*np.sqrt(b2)*atan(a1/np.sqrt(b2)) +b1*np.sqrt(a2)*atan(b1/np.sqrt(a2))-a1*atan(a1)-b1*atan(b1))
    hr=sig*((T_r+273.5)**(2)+(T[0]+273.5)**(2))*((T_r+273.5)+(T[0]+273.5))/(1/esp_p+1/esp_r-2+1/F_sp)
    hw=1.4*10**(-3)*T[0]+0.27*W[0]-4.0*10**(-4)*T[0]*W[0]-0.77*W[0]**(2)
    temp=lam*(170+284*W[0])*Dw*hw;
    #print "temp: ", temp;
    T_f=T[1]+2*dx/k*(hr*(T_r-T[0])+hc*(T_air-T[0])-temp*(W[0]-W_air))
    w_f=W[1]-2*dx*hw*(W[0]-W_air)
    #print "T_f, w_f",T_f, w_f,0
    r=k*dt/((170+284*W[0])*cp*dx*dx);
    a[0,0]=1+2*r*(1-theta)*(1+dx*hr/k+dx*hc/k)
    a[0,1]=-2*r*(1-theta)
    b[0]=r*theta*T_f+(1-2*r*theta)*T[0]+r*theta*T[1]+lam*Dw*dt/(cp*dx*dx)*(W[1]-2*W[0]+w_f)+r*(1-theta)*2*(dx/k)*(hr*T_r+hc*T_air-temp*(W[0]-W_air))
    #print "b[0]",b[0]
    #print "ALPHA3,4: r*theta", r*theta
    #print "(1-2*r*theta)*T[0]",(1-2*r*theta)*T[0]
    #print "ALPHA3,4*T[1,j], T[i,1] -- r*theta*T[1]",r*theta*T[1]
    #print "lam*Dw*dt/(cp*dx*dx)",lam*Dw*dt/(cp*dx*dx)
    #print "(W[1]-2*W[0]+w_f)",(W[1]-2*W[0]+w_f)
    #print "r*(1-theta)*2*(dx/k)",r*(1-theta)*2*(dx/k)
    #print "hr*T_r+hc*T_air",hr*T_r+hc*T_air
    #print "temp*(W[0]-W_air)",temp*(W[0]-W_air)
    
    
    #print 
    #exit()
    #**************************************
    #for Temp at last node
    #**************************************
    T[N+1]=T[N-1]
    r=k*dt/((170+284*W[N])*cp*dx*dx)
    a[N,N-1]=-2*r*(1-theta)
    a[N,N]=1+2*r*(1-theta)
    b[N]=r*theta*T[N-1]+(1-2*r*theta)*T[N]+r*theta*T[N-1]+lam*Dw*dt/(cp*dx*dx)*(W[N-1]-2*W[N]+W[N+1])

    return np.linalg.solve(a,b)


################################################
#Function to correct vapour and water content.
################################################
def correction(T_new,V,W,N):
    R=8.314
    #********************************
    # data points for interploation
    #********************************
    x=range(0,100+1,2)
    y=[.611, .705, .813, .934, 1.072, 1.226, 1.401, 1.597, 1.817, 2.062, 2.337, 2.642, 2.983, 3.360, 3.779, 4.242, 4.755, 5.319, 5.941, 6.625, 7.377, 8.201, 9.102, 10.087, 11.164, 12.34, 13.61, 15., 16.5, 18.14, 19.92, 21.83, 23.9, 26.14, 28.55, 31.15, 33.94, 36.95, 40.18, 43.63, 47.33, 51.31, 55.56, 60.11, 64.93, 70.09, 75.58, 81.43, 87.66, 94.28, 101.31]
    x=np.hstack((x,range(105,180+1,5) ))
    y=np.hstack((y, [120.82, 143.27, 169.06, 198.53, 232.1, 270.1, 313., 361.2, 415.4, 475.8, 543.1, 617.8, 700.5, 791.7, 892.0, 1002.1]))
    x=np.hstack((x, [190, 200, 225, 250, 275, 300]))
    y=np.hstack((y, [1254.4, 1553.8, 2548, 3973, 5942, 8581]))
    #************************************************************
    # interpolation and calculation of saturated amount of vapor
    #************************************************************

    P = np.zeros((N+2))
    V_s = np.zeros((N+2))
    V_temp = np.zeros((N+2))
    W_temp = np.zeros((N+2))

    for i in range(0,N+1):
        f=interpolate.interp1d(x,y) #interp1(x,y,T_new[i],'spline')*1000
        P[i] = f(T_new[i])*1000
        V_s[i]=18.*10**(-3)*P[i]/(R*(T_new[i]+273.5)*(170+281*W[i]))*0.7*3.8
    #****************************************
    # correction in vapour and water content
    #****************************************
    for i in range(0,N+1):
        if (W[i]+V[i]<V_s[i]):
            V_temp[i]=W[i]+V[i]
            W_temp[i]=0
        else:
            V_temp[i]=V_s[i]
            W_temp[i]=W[i]+V[i]-V_s[i]

    return V_temp,W_temp,V_s,P

####################################
# Function to find new Vapour
####################################

def Vnew(T_new,V_temp,W_temp,dx,dt,N,theta):
    a=np.zeros((N+1,N+1))
    b=np.zeros((N+1))
    V_air=0
    #**********************************
    # V at internal points
    #**********************************
    for i in range(1,N):# i=2:1:N
        r=dt*9.0*10**(-12)*(T_new[i]+273.5)**(2)/(dx*dx)
        a[i,i-1]=-r*(1-theta)
        a[i,i]=1+2*r*(1-theta)
        a[i,i+1]=-r*(1-theta)
        b[i]=r*theta*V_temp[i-1]+(1-2*r*theta)*V_temp[i]+r*theta*V_temp[i+1]
    #*************************
    # V at 1st boundary
    #*************************
    temp=2*dx*3.2*10**(9)/((T_new[0]+273.5)**(3))
    r=dt*9.0*10**(-12)*(T_new[0]+273.5)**(2)/(dx*dx)
    V_f=V_temp[1]-temp*(V_temp[0]-V_air)
    a[0,0]=1+r*(1-theta)*(2+temp)
    a[0,1]=-2*r*(1-theta)
    b[0]=r*theta*V_f+(1-2*r*theta)*V_temp[0]+r*theta*V_temp[1]+temp*r*(1-theta)*V_air
    #**************************
    #V at last boundary
    #**************************
    V_temp[N+1]=V_temp[N-1]
    r=dt*9.0*10**(-12)*(T_new[N]+273.5)**(2)/(dx*dx)
    a[N,N-1]=-2*r*(1-theta)
    a[N,N]=1+2*r*(1-theta)
    b[N]=r*theta*V_temp[N-1]+(1-2*r*theta)*V_temp[N]+r*theta*V_temp[N+1]
    #********************
    #solving
    #*********************
    return np.linalg.solve(a,b)

##############################################
#second correction of vapour and water content.
##############################################
def Correction2(T_new,V_new,W_temp,V_s,N,P,W):
    R=8.314
    for i in range(0,N+1):#i=1:1:N+1
        V_s[i]=18.*10**(-3)*P[i]/(R*(T_new[i]+273.5)*(170+281*W[i]))*0.7*3.8

    for i in range(0,N+1):
        if (W_temp[i]+V_new[i]<V_s[i]):
            V_new[i]=W_temp[i]+V_new[i]
            W_temp[i]=0
        else:
            W_temp[i]=W_temp[i]+V_new[i]-V_s[i]
            V_new[i]=V_s[i]

    return V_new,W_temp

##########################################
#Function to calculate new water content.
##########################################
def Wnew(T_new,V_new,W_temp,dx,dt,N,theta):
    a=np.zeros((N+1,N+1))
    b=np.zeros((N+1))
    W_air=0
    Dw=1.35*10**(-10)
    #*******************************
    #Internal nodes
    #*******************************
    for i in range(1,N):#i=2:1:N:
        r=dt*Dw/(dx*dx)
        a[i,i-1]=-r*(1-theta)
        a[i,i]=1+2*r*(1-theta)
        a[i,i+1]=-r*(1-theta)
        b[i]=r*theta*W_temp[i-1]+(1-2*r*theta)*W_temp[i]+r*theta*W_temp[i+1]
    #******************************
    # W at 1st boundary
    #******************************
    temp=2*dx*(1.4*10**(-3)*T_new[0]+0.27*W_temp[0]-4.0*10**(-4)*T_new[0]*W_temp[0]-0.77*W_temp[0]*W_temp[0])
    w_f=W_temp[1]-temp*(W_temp[0]-W_air)
    r=dt*Dw/(dx*dx)
    a[0,0]=1+r*(1-theta)*(2+temp)
    a[0,1]=-2*r*(1-theta)
    b[0]=r*theta*w_f+(1-2*r*theta)*W_temp[0]+r*theta*W_temp[1]+r*(1-theta)*temp*W_air
    #*****************************
    #W at last boundary
    #*****************************
    W_temp[N+1]=W_temp[N-1]
    r=dt*Dw/(dx*dx)
    a[N,N-1]=-2*r*(1-theta)
    a[N,N]=1+2*r*(1-theta)
    b[N]=r*theta*W_temp[N-1]+(1-2*r*theta)*W_temp[N]+r*theta*W_temp[N+1]

    return np.linalg.solve(a,b)
 
    ########### END ##########


