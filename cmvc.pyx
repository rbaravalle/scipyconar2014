import numpy as np
cimport numpy as np


DTYPE = np.uint8

ctypedef np.uint8_t DTYPE_t
ctypedef np.int32_t DTYPE_ti
ctypedef np.float32_t DTYPE_tf

def cmvc(np.ndarray[DTYPE_t, ndim=2] field,int N,np.ndarray[DTYPE_ti, ndim=1] xn):
    cdef np.ndarray[DTYPE_t, ndim=2] field2 = np.zeros((N,N),dtype=DTYPE)

    cdef int x,y, a,b, xxn
    for x from 0<=x<N:
        for y from 0<=y<N:
           xxn = xn[x+y*N]
           a = xxn%N
           b = xxn/N
           field2[x,y] = field[a,b]

    return field2
