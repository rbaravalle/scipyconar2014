from Tkinter import *
import ImageTk, Image
import numpy as np
import pyopencl as cl
import time
import fractalBread3DCL2 as fractal
import cmvc

# INTERACTIVE MEAN VALUE COORDINATES ON THE GPU AND CYTHON
root = Tk()

# inherites from Tkinter.Tk
class DragApp():
    '''Drag and drop class'''

    def __init__(self, *args, **kwargs):

        self.clprog()
        # create a canvas
        self.canvas = Canvas(root, width=800, height=800, background="#000000")

        self.img = Image.open('/home/rodrigo/bakedslice14.png').convert("L")
        data = np.array(self.img.getdata()).astype(np.uint8)
        self.field = data.reshape(self.img.size[0], self.img.size[1])
        self.imgTk = ImageTk.PhotoImage(self.img)
        self.N = self.imgTk.width()

        self.cageReal,self.cageNew,self.cageOrig,_ = self.computeCages()


        # quit button
        self.button = Button(root, text = "Save", command = self.compute, anchor = 'w',
                            width = 10, activebackground = "#33B5E5")
        self.quit_button_window = self.canvas.create_window(750, 0, anchor='nw', window=self.button)    


        self.image_on_canvas = self.canvas.create_image(0,0,image=self.imgTk,anchor="nw")

        # this data is used to keep track of an 
        # item being dragged
        self._drag_data = {"x": 0, "y": 0, "item": None}


        # original, for reference
        map(lambda i: self._create_object((i[1], i[0]), "Red"), self.cageOrig)



        self.lines()

        # add bindings for clicking, dragging and releasing over
        # any object with the "token" tag
        self.canvas.tag_bind("token", "<ButtonPress-1>", self.OnTokenButtonPress)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self.OnTokenButtonRelease)
        self.canvas.tag_bind("token", "<B1-Motion>", self.OnTokenMotion)


        self.button2 = Button(root, text = "Quit", command = self.out, anchor = 'w',
                            width = 10, activebackground = "#33B5E5")
        self.quit2_button_window = self.canvas.create_window(750, 35, anchor='nw', window=self.button2)  

        # create a couple movable objects
        self.firstMovable = self._create_token((self.cageOrig[0][1], self.cageOrig[0][0]), "Green")
        for i in range(1,len(self.cageOrig)):
            self._create_token((self.cageOrig[i][1], self.cageOrig[i][0]), "Green")
        #map(lambda i: self._create_token((i[1], i[0]), "Green"), self.cageOrig)

        self.canvas.pack()

    def compute(self):
        # call fractalBread with cages
        self.canvas.postscript(file="pepe.ps", colormode='color', x = 100, y = 100, width = self.N-200, height = self.N - 200)
        fractal.main(1,0.15,2.78,2,14,self.cageOrig,self.cageReal)

    def out(self):
        exit()

    def lines(self):
        self.line = []
        for i in range(len(self.cageNew)-1):
            self.line.append(self.canvas.create_line(self.cageNew[i][1], self.cageNew[i][0], self.cageNew[i+1][1], self.cageNew[i+1][0],fill="Green"))

        self.line.append(self.canvas.create_line(self.cageNew[i+1][1], self.cageNew[i+1][0], self.cageNew[0][1], self.cageNew[0][0],fill="Green"))

        self.lineOrig = []
        for i in range(len(self.cageOrig)-1):
            print i
            self.lineOrig.append(self.canvas.create_line(self.cageOrig[i][1], self.cageOrig[i][0], self.cageOrig[i+1][1], self.cageOrig[i+1][0],fill="Red"))

        self.lineOrig.append(self.canvas.create_line(self.cageOrig[i+1][1], self.cageOrig[i+1][0], self.cageOrig[0][1], self.cageOrig[0][0],fill="Red"))

    def lines2(self):
        print self.cageNew
        for i in range(len(self.cageNew)-1):
            print self.line[i]
            self.canvas.coords(self.line[i],self.cageNew[i][1], self.cageNew[i][0], self.cageNew[i+1][1], self.cageNew[i+1][0])
            root.update_idletasks() # redraw
            root.update() # process events

        self.canvas.coords(self.line[i+1],self.cageNew[i+1][1], self.cageNew[i+1][0], self.cageNew[0][1], self.cageNew[0][0])
        root.update_idletasks() # redraw
        root.update() # process events



    def _create_token(self, coord, color):
        '''Create a token at the given coordinate in the given color'''
        (x,y) = coord
        return self.canvas.create_oval(x-8, y-8, x+8, y+8, 
                                outline=color, fill=color, tags="token")

    def _create_object(self, coord, color):
        '''Create a oval at the given coordinate in the given color'''
        (x,y) = coord
        self.canvas.create_oval(x-8, y-8, x+8, y+8, 
                                outline=color, fill=color)

    def OnTokenButtonPress(self, event):
        '''Being drag of an object'''
        # record the item and its location
        self.ind = self.canvas.find_closest(event.x, event.y,start=self.firstMovable)[0]
        self._drag_data["item"] = self.ind

        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self.delta_x = 0
        self.delta_y = 0



    def OnTokenButtonRelease(self, event):
        '''End drag of an object'''

        # Refresh new positions
        self.cageReal[self._drag_data["item"]-self.firstMovable][0] -= self.delta_y
        self.cageReal[self._drag_data["item"]-self.firstMovable][1] -= self.delta_x

        self.cageNew[self._drag_data["item"]-self.firstMovable][0] += self.delta_y
        self.cageNew[self._drag_data["item"]-self.firstMovable][1] += self.delta_x

        # Compute New Warp
        #p=int(N/4)
        eps = 10.0*np.nextafter(0,1)

        nSize = len(self.cageOrig)

        a = np.array(self.cageOrig).astype(np.int32)
        cageOrig_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a)

        a = np.array(self.cageReal).astype(np.int32)
        cageReal_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a)

        s = np.zeros((nSize,2)).astype(np.float32)
        s_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a)

        p = 0
        t = time.clock()

        xn = np.zeros((self.N*self.N)).astype(np.int32)
        d2 = np.zeros((nSize)).astype(np.float32)
        d = np.zeros((nSize)).astype(np.float32)
        dest_xn_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, xn.nbytes)

        self.prg.main(self.queue, (self.N,self.N), None, cageOrig_buf, cageReal_buf, dest_xn_buf, np.int32(nSize), np.float32(eps), np.int32(p), np.int32(self.N))
        cl.enqueue_read_buffer(self.queue, dest_xn_buf, xn).wait()

        print "TIEMPO MVC: ", time.clock()-t
        field2 = cmvc.cmvc(self.field,self.N,xn)

        print "TIEMPO MVC: ", time.clock()-t
        self.img = Image.frombuffer('L',(self.N,self.N), np.array(field2).astype(np.uint8),'raw','L',0,1)

        self.imgTk = ImageTk.PhotoImage(self.img)

        # update image!
        self.canvas.itemconfig(self.image_on_canvas, image = self.imgTk)

        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        self.delta_x = 0
        self.delta_y = 0

        self.lines2()


    def OnTokenMotion(self, event):
        '''Handle dragging of an object'''
        # compute how much this object has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        self.delta_x += delta_x
        self.delta_y += delta_y
        # move the object the appropriate amount
        if(self.ind>=self.firstMovable):
            self.canvas.move(self._drag_data["item"], delta_x, delta_y)
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def computeCages(self):

        N = self.N

        #p = 128
        # Arbitrary shape, mean value coordinates
        #cageOrig = np.array([[p,N-1-p],[N/2,N-1-p],[N/2+40,N-1-p],[N-1-p,N-1-p],[N-1-p,N/2+40],[N-1-p,N/2],[N-1-p,N/2-40],[N-1-p,p],[N/2+40,p],[N/2,p],[N/2-40,p],[p,p]]).astype(np.float32)

        #cageReal = np.array(cageOrig)
        #cageNew = np.array(cageOrig)

        # control points displacements
        # RIGHT, BOTTOM,LEFT, TOP < |> <_> <| > <->
        # X: BOTTOM - TOP
        # Y : LEFT - RIGHT
        #trs=[[0,-20],[0,0],[0,0],[0,0],[30,0],[45,0],[30,40],[-5,0],[30,10],[25,10],[45,20],[12,15]]
        #trs=[[0,-10],[0,0],[0,0],[0,0],[20,0],[35,0],[20,30],[-5,0],[20,10],[15,10],[25,10],[12,5]]

    #    for i in range(len(cageOrig)):
    #        cageReal[i] = cageOrig[i]+trs[i]
    #        cageNew[i] = cageOrig[i]-trs[i]


    #    return cageReal, np.array(map (lambda i: np.array(i),cageNew)), cageOrig

        p = int(0)

        # Arbitrary shape, mean value coordinates
        #cageOrig = np.array([[p,N-1-p],[N/2,N-1-p],[N/2+40,N-1-p],[N-1-p,N-1-p],[N-1-p,N/2+40],[N-1-p,N/2],[N-1-p,N/2-40],[N-1-p,p],[N/2+40,p],[N/2,p],[N/2-40,p],[p,p]]).astype(np.float32)
        cageOrig = np.array([[p,N-1-p],[p+50,N-1-p],[N/2,N-1-p],[N/2+50,N-1-p],[N-1-p,N-1-p],[N-1-p,N/2+50],[N-1-p,N/2],[N-1-p,N/2-50],[N-1-p,p],[N/2+50,p],[N/2,p],[N/2-50,p],[p,p],[p,N/2-50],[p,N/2],[p,N/2+50]]).astype(np.float32)

        cageReal = np.array(cageOrig)
        cageNew = np.array(cageOrig)

        cageReal[:] = cageOrig[:]
        cageNew[:] = cageOrig[:]

        return cageReal, np.array(map (lambda i: np.array(i),cageNew)), cageOrig,p

    def clprog(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

        self.prg = cl.Program(self.ctx, """
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


if __name__ == "__main__":
    app = DragApp()
    root.mainloop()
