
# third-party libraries
import numpy as np
import scipy.io

from fig import field

# parameters
folders = "raw/vel_"    # data folder
nsteps = 800            # number of timesteps in each case
ncases = 14             # number of cases
nx = 80                 # number of cells in x
ny = 40                 # number of cells in y
nt = 20                 # number of steps (e.g., 10 for input + 10 for prediction)

# solution arrays
fi = np.zeros((int(nsteps*ncases/nt),nx,ny,nt))
u = np.zeros((int(nsteps*ncases/nt),nx,ny,nt))
#x = np.zeros((int(nsteps*ncases/nt),nx,ny,nt))
#y = np.zeros((int(nsteps*ncases/nt),nx,ny,nt))
print("shape: ", u.shape, flush=True)

# loop over cases
batch_count = 0
step_count  = 0
for i in range(0,ncases):
  print("Reading case ", i+1, flush=True)

  # loop over time steps
  for j in range(0,nsteps):
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
    data = np.loadtxt(folders + str(i+1) + '/fi_' + num + '.csv', delimiter=',', skiprows=1, usecols=(0,1,2,3))
    fi_j = np.reshape(data[:,0], (nx,ny))
    u_j  = np.reshape(data[:,1], (nx,ny))

    if i==0 and j==0:
      x = np.unique(data[:,2])
      y = np.unique(data[:,3])

      x_c  = np.ndarray.flatten(np.meshgrid(y, x)[1])
      y_c  = np.ndarray.flatten(np.meshgrid(y, x)[0])
      field(x_c, y_c, np.ndarray.flatten(fi_j), str(step_count) + str(batch_count) + ' phi')
      #field(x_c, y_c, np.ndarray.flatten(u_j), str(step_count) + str(batch_count)  + ' vel')

    #field(data[:,3], data[:,4], data[:,1], str(step_count) + str(batch_count))
    
    # merge datasets
    fi[batch_count,:,:,step_count] = fi_j
    u[batch_count,:,:,step_count] = u_j
    #x[batch_count,:,:,step_count] = x_j
    #y[batch_count,:,:,step_count] = y_j

    # adjust counts
    step_count = step_count + 1
    if step_count == nt:
      step_count = 0
      batch_count = batch_count + 1
      print("Batch count:", batch_count-1)

field(x_c, y_c, np.ndarray.flatten(fi[0,:,:,10]), str(step_count) + str(batch_count) + 'phi')
#field(x_c, y_c, np.ndarray.flatten(u[0,:,:,10]),  str(step_count) + str(batch_count) + ' vel')

# save data
print("Saving data")
scipy.io.savemat('data.mat', mdict={'u': u, 'x': x, 'y': y, 'fi': fi})