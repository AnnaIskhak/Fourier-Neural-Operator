
# third-party libraries
import numpy as np
import scipy.io

from fig import field

# parameters
folders = "raw/vel_"    # data folder
nx = 80                 # number of cells in x
ny = 40                # number of cells in y
nt = 800                # number of steps
case = 15               # 16th case for a 5/bubble diameter channel

# solution arrays
fi = np.zeros((1,nx,ny,nt))
u = np.zeros((1,nx,ny,nt))
#x = np.zeros((1,nx,ny,nt))
#y = np.zeros((1,nx,ny,nt))

# loop over time steps
step_count  = 0
for j in range(0,nt):
    #print("Reading timestep", j+1, flush=True)

    # time step number
    num = str(j)

    #if (j+1 < 10) and (j+1 >= 0):
    #    num = '000' + str(j+1)
    #if (j+1 < 100) and (j+1 >= 10):
    #    num = '00' + str(j+1)
    #if (j+1 < 1000) and (j+1 >= 100):
    #    num = '0' + str(j+1)

    # read data
    data = np.loadtxt(folders + str(case) + '/fi_' + num + '.csv', delimiter=',', skiprows=1, usecols=(0,1,2,3))
    fi_j = np.reshape(data[:,0], (nx,ny))
    u_j = np.reshape(data[:,1], (nx,ny))    
    #x_j = np.reshape(data[:,3], (nx,ny))
    #y_j = np.reshape(data[:,4], (nx,ny))
    if j==0:
        x = np.unique(data[:,2])
        y = np.unique(data[:,3])

        x_c  = np.ndarray.flatten(np.meshgrid(y, x)[1])
        y_c  = np.ndarray.flatten(np.meshgrid(y, x)[0])
        field(x_c, y_c, np.ndarray.flatten(fi_j), str(step_count) + str(0) + ' phi')
    #field(data[:,3], data[:,4], data[:,1], str(step_count) + str(batch_count))
    
    # merge datasets
    fi[0,:,:,step_count] = fi_j
    u[0,:,:,step_count] = u_j
    #x[0,:,:,step_count] = x_j
    #y[0,:,:,step_count] = y_j

    # adjust counts
    step_count = step_count + 1

#field(np.ndarray.flatten(x[0,:,:,10]), np.ndarray.flatten(y[0,:,:,10]), np.ndarray.flatten(u[0,:,:,10]), str(step_count) + str(batch_count))

# save data
print("Saving data")
scipy.io.savemat('data_pred.mat', mdict={'u': u, 'x': x, 'y': y, 'fi': fi})