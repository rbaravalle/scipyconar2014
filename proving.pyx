import numpy as np
cimport numpy as np

from libc.stdlib cimport rand

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

cdef extern from "limits.h":
    int INT_MAX

cdef extern from "math.h":
    float pow(int x ,float y)
    int floor(float x)

def proving(int param_a,float param_b,float param_c,int param_d,int param_e, int N, int Nz):
    cdef int r, v, i, j, k,x,y,z, maxrank
    cdef float cubr
    cdef np.ndarray[DTYPE_t, ndim=3] field = np.zeros((N,N,Nz),dtype=DTYPE) + np.uint8(255)
    cubr = (param_b/float(20.0))*N*N*Nz
    for r from param_d <= r < param_e by param_a:
        maxrank = floor(cubr/(pow(r,param_c)))
        if(maxrank >=1.0):
            for v from 0<=v< maxrank:
                x = floor((rand() / float(INT_MAX))*N)
                y = floor((rand() / float(INT_MAX))*N)
                z = floor((rand() / float(INT_MAX))*Nz)
                for i from x-r<=i<x+r:
                    for j from y-r<=j<y+r:
                        for k from z-r<=k<z+r:
                            if((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k) < r*r):
                                if(i < N and i >= 0 and j < N and j >= 0 and k < Nz and k >= 0 ):
                                    field[i,j,k] = 0

    return field
