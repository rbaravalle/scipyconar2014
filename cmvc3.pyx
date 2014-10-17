import numpy as np
cimport numpy as np


DTYPE = np.uint8

ctypedef np.uint8_t DTYPE_t
ctypedef np.int32_t DTYPE_ti
ctypedef np.float32_t DTYPE_tf

def cmvc(np.ndarray[DTYPE_t, ndim=3] field2,int N,np.ndarray[DTYPE_ti, ndim=1] xn, int Nz, int p):
    cdef np.ndarray[DTYPE_t, ndim=3] field3 = np.zeros((N,N,Nz),dtype=DTYPE)

    cdef int x,y, a,b, xxn
    for x from p<=x<N-p:
        for y from p<=y<N-p:
           xxn = xn[x+y*(N-2*p)]
           a = xxn%(N-2*p)
           b = xxn/(N-2*p)
           field3[x,y] = field2[a+p,b+p]

    return field3
