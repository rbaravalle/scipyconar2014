import os
from subprocess import call
import time
import sys

arr = [ "cmvc", "cmvc3","proving", "cbaking"]

for i in range(len(arr)):

    command1 = "cython "+arr[i]+".pyx "
    command2 = "gcc -c -fPIC -I/usr/include/python2.7/ "+arr[i]+".c"
    command3 = "gcc -shared "+arr[i]+".o -o "+arr[i]+".so"

    print command1
    os.system(command1)
    print command2
    os.system(command2)
    print command3
    os.system(command3)

