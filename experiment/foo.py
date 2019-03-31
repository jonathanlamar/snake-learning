from time import sleep
import os
import numpy as np

# Proof of concept
#for x in range(50):
#    print('\rDownloading: {0:s} ({1:d})'.format('|'*(x//2), x), end='')
#    sys.stdout.flush()
#    sleep(0.1)

def build_board():
    A = np.zeros((10,10))
    i,j = np.random.choice(10, 2, replace=True)
    A[i,j] = 1
    B = [['0' if A[i,j]==0 else '1' for j in range(10)] for i in range(10)]
    return B

for x in range(10):
    os.system('clear')
    B = build_board()
    for i in range(10):
            for j in range(10):
                print('\033[{0};{1}H{2}'.format(i,j,B[i][j]))
    sleep(0.1)
