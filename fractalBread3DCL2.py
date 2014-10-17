import numpy as np
import random
import Image
import ImageDraw
import os
import time


from baking1D import calc
#from mvc import mvc # mean value coordinates
import pyopencl as cl
import csv

# Cython
import proving
import cbaking
import cmvc3

N = 356
Nz = 100

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

prg = cl.Program(ctx, """
    __kernel void main( __global int *cageOrig, __global int *cageNew, __global int *xn, const int nSize, const float eps, const int p, const int N) {
        int x = get_global_id(0)+p;
        int y = get_global_id(1)+p;

        float dest[17], dest2[17], dx, dy;
        int s[17*2];

        int i,h;

        for(i = 0; i < nSize; i++) {
            dest[i]   =   0;
            dest2[i]  =   0;
            s[2*i]    =   cageOrig[2*i]   - x;
            s[2*i+1]  =   cageOrig[2*i+1] - y;
        }

        int cut = 0; // FIX ME!!: we should skip one part or not 
        for(i = 0; i < nSize; i++) {
            dx  =   (float)s[2*i];
            dy  =   (float)s[2*i+1];
            int ip = (i+1)%nSize;
            float sC = s[2*ip];
            float sD = s[2*ip+1];

            float ri = sqrt( dx*dx + dy*dy );
            float Ai = 0.5*(dx*sD - sC*dy);
            float Di = sC*dx + sD*dy;


            if(ri <= eps) {dest[i] = 1.0; cut = 1; break;}
            else {
                if( fabs((float)Ai) == 0.0 && Di < 0.0){
                    dx = (float)(cageOrig[2*ip] - cageOrig[2*i]);
                    dy = (float)(cageOrig[2*ip+1] - cageOrig[2*i+1]);
                    float dl = sqrt( dx*dx + dy*dy);
                    dx = (float)(x - cageOrig[2*i]);
                    dy = (float)(y - cageOrig[2*i+1]);
                    float mu = sqrt(dx*dx + dy*dy)/dl;
                    dest[i]  = 1.0-mu;
                    dest[ip] = mu;
                    cut = 1;
                    break;
                }
            }

            float rp = sqrt( sC*sC + sD*sD );
            if(Ai == 0.0) dest2[i] = 0.0;
            else dest2[i] = (ri*rp - Di)/(2.0*Ai);
        }

        if(cut == 0) {
            float wsum = 0.0;
            for(i = 0; i < nSize; i++) {
                dx  =   (float)(cageOrig[2*i]   - x);
                dy  =   (float)(cageOrig[2*i+1] - y);
                float ri = sqrt( dx*dx + dy*dy );
                int im = (nSize-1+i)%nSize;
                float wi = 2.0*( dest2[i] + dest2[im] )/ri;
                dest[i] = wi;
                wsum += wi;
            }


            if( fabs(wsum) > 0.0)
                for(i = 0; i < nSize; i++) 
                    dest[i] /= wsum;
        }

        // dest : computed barycoords
        float msumx = 0.0;
        float msumy = 0.0;
        for(i = 0; i < nSize; i++) {
            msumx += dest[i]*cageNew[2*i];
            msumy += dest[i]*cageNew[2*i+1];
        }

        int isumx = (int)msumx;
        int isumy = (int)msumy;

        if(isumx > N-1) isumx = N-1;
        if(isumy > N-1) isumy = N-1;
        if(isumx < 0) isumx = 0;
        if(isumy < 0) isumy = 0;

        xn[x+y*(N-2*p)] = isumx+isumy*(N-2*p);

    }
    """).build()


def readCSV(where):
    arr = np.zeros((N+1,N+1)).astype(np.float32)
    with open(where, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            arr[i] = row
            i = i+1
    return arr


def main(param_a,param_b,param_c,param_d,param_e,cageOrig,cageNew):

    loadCSV = False
    field2 = np.zeros((N,N,Nz)).astype(np.uint8)
    field3 = np.zeros((N,N,Nz)).astype(np.uint8)
    Ires = 0
    pp = 50
    I = Image.new('L',(N,Nz*N),0.0)
    I3 = Image.new('L',(N-2*pp,(N-2*pp)*(Nz)),0.0)
    t2 = t1 = t3= 0
    p=int(N/4)
    eps = 10.0*np.nextafter(0,1)
    nSize = len(cageOrig)
    k = float(10.0)
    cageNew = np.array(map (lambda i: np.array(i),cageNew))

    if not os.path.isdir('warp2'): 
        os.mkdir ( 'warp2' ) 

    if not os.path.isdir('warp2/baked'): 
        os.mkdir ( 'warp2/baked' ) 

    if not os.path.isdir('warp2/warped'): 
        os.mkdir ( 'warp2/warped' ) 

    print "Proving..."
    t = time.clock()
    field = proving.proving(param_a,param_b,param_c,param_d,param_e,N,Nz)
    #import poisson3D
    #field = poisson3D.main()
    print "Proving Time: ", time.clock()-t

    if(loadCSV):
        arr = readCSV('exps/baking.csv')
    else:
        arr = calc()

    gx, gy = np.gradient(arr)

    print "Proving..."
    t = time.clock()
    field2 = cbaking.cbaking(field,N,k,gx,gy,Nz)
    print "Baking Time: ", time.clock()-t

    print "Warping..."
    a = np.array(cageOrig).astype(np.int32)
    cageOrig_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

    a = np.array(cageNew).astype(np.int32)
    cageNew_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

    s = np.zeros((nSize,2)).astype(np.float32)
    s_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

    p = 0

    xn = np.zeros(((N-2*p)*(N-2*p))).astype(np.int32)
    d2 = np.zeros((nSize)).astype(np.float32)
    d = np.zeros((nSize)).astype(np.float32)
    dest_xn_buf = cl.Buffer(ctx, mf.WRITE_ONLY, xn.nbytes)

    t = time.clock()
    prg.main(queue, (N-2*p,N-2*p), None, cageOrig_buf, cageNew_buf, dest_xn_buf, np.int32(nSize), np.float32(eps), np.int32(p), np.int32(N))
    cl.enqueue_read_buffer(queue, dest_xn_buf, xn).wait()
    field3 = cmvc3.cmvc(field2,N,xn,Nz,p)
    print "Warping time: ", time.clock()-t

    print "Saving Image..."
    Ires = 0


    for w in range(Nz):
        II = Image.frombuffer('L',(N-2*pp,N-2*pp), np.array(field3[pp:N-pp,pp:N-pp,w]).astype(np.uint8),'raw','L',0,1)
        II.save('warp2/warped/warpedslice'+str(w)+'.png')
        I3.paste(II,(0,(N-2*pp)*w))

    I3.save('warp2/warped.png')
    print "Image warp2/warped.png saved"
    return Ires,pp,k


#main(1,0.06,2.7,2,20)
