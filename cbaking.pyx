import numpy as np
cimport numpy as np


DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

cdef extern from "math.h":
    float sqrt(float x)
    float round(float x)

def cbaking(np.ndarray[DTYPE_t, ndim=3] field,int N,float k,np.ndarray[DTYPE_tf, ndim=2] gx,np.ndarray[DTYPE_tf, ndim=2] gy, int Nz):
    cdef np.ndarray[DTYPE_t, ndim=3] field2 = np.zeros((N,N,Nz),dtype=DTYPE)

    cdef float u,v,dist
    cdef int x,y
    for x from 0<=x <N:    
        for y from 0<=y <N:
            dist = sqrt(((x-N/2)*(x-N/2)+(y-N/2)*(y-N/2)))
            u = round(x+k*gx[x,y]*dist)
            v = round(y+k*gy[x,y]*dist)
            if(u >= 0 and u < N and v >= 0 and v < N):
                field2[x,y] = field[u,v]

    return field2
