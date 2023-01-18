# third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# font, size, ... for figures
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['lines.markersize'] = 1.0
plt.rcParams.update({'errorbar.capsize': 3})
set_dpi = 600

def Plot_Loss(epochs, q_valid, q_train, lr_hist, name):

    print("   Making loss plot for " + name, flush=True)
    x = np.arange(epochs)
    fig, ax = plt.subplots()
    ax.plot(x, q_valid, color='green', label='validation', linewidth=1)
    ax.plot(x, q_train, color='black', label='training',   linewidth=1)

    # show the point with minimal validation loss
    q_valid_min = np.min(q_valid)
    ind_q_valid_min = np.where(q_valid == q_valid_min)
    ax.plot(x[ind_q_valid_min], q_valid_min, 'ro', label='saved model', markersize=5)

    ax.set_xlabel('epoch')
    ax.set_ylabel(name + ' loss')
    ax.set_yscale('log')
    ax.legend(loc='best', markerscale=1, frameon=False, prop={'size': 16})
    #plt.grid()

    # plot lr history
    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Learning rate', color=color)
    ax2.plot(x, lr_hist, color = color)
    ax2.tick_params(axis ='y', labelcolor = color)

    plt.tight_layout()
    plt.savefig('work/loss_' + name + '.jpg', dpi=set_dpi, format='jpg', bbox_inches='tight')
    #plt.show()

def Plot_45(q, q_data, name):
    """
    45-degree plot
    """
    print("Making 45 degree plots", flush=True)
    fig, ax = plt.subplots()
    ax.plot(q_data, q_data, color='green', label='data', linewidth=2)
    ax.scatter(q, q_data, s=50, c='red', marker='o', edgecolors='black', label='NN',  )
    ax.set_ylabel('data')
    ax.set_xlabel('predictions')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', markerscale=1, frameon=False, prop={'size': 16})
    plt.tight_layout()
    plt.savefig('work/' + name + '.jpg', dpi=set_dpi, format='jpg', bbox_inches='tight')
    #plt.show()

# make 2D field
def Field(x, y, z1, t1, title):

    print("    Making field plots", flush=True)

    plt.rcParams['font.size'] = 14

    #
    fig, ([ax1, ax2, ax3]) = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(20,3)
    #
    ax1.tricontour(x, y, z1)
    cntr1 = ax1.tricontourf(x, y, z1, 10, cmap="rainbow")
    fig.colorbar(cntr1, ax=ax1)
    ax1.set(xlim=(np.amin(x), np.amax(x)), ylim=(np.amin(y), np.amax(y)))
    ax1.set_title("prediction")
    ax1.set_aspect('equal')
    #
    ax2.tricontour(x, y, t1)
    cntr2 = ax2.tricontourf(x, y, t1, 10, cmap="rainbow")
    fig.colorbar(cntr2, ax=ax2)
    ax2.set(xlim=(np.amin(x), np.amax(x)), ylim=(np.amin(y), np.amax(y)))
    ax2.set_title("true")
    ax2.set_aspect('equal')
    #
    ax3.tricontour(x, y, t1-z1)
    cntr3 = ax3.tricontourf(x, y, t1-z1, 10, cmap="rainbow")
    fig.colorbar(cntr3, ax=ax3)
    ax3.set(xlim=(np.amin(x), np.amax(x)), ylim=(np.amin(y), np.amax(y)))
    ax3.set_title("error")
    ax3.set_aspect('equal')
    #
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    plt.savefig('work/' + title + '.jpg', dpi=set_dpi, format='jpg', bbox_inches='tight')
    #plt.show()

def Plot_1D(x, z1, t1, title):

    print("   Making 1D plot for " + title, flush=True)
    #x = np.arange(x)
    fig, ax = plt.subplots()
    #ax.plot(x, z1, color='green', label='prediction', linewidth=1)
    
    ax.plot(x, t1-z1, color='black', linewidth=1)
    ax.set_xlabel('m')
    ax.set_ylabel(' absolute error at the centerline, m')
    #ax.set_yscale('log')
    #ax.legend(loc='best', markerscale=1, frameon=False, prop={'size': 16})
    #plt.grid()
    ax.set_xlim(np.amin(x), np.amax(x))
    plt.tight_layout()
    plt.savefig('work/error_' + title + '.jpg', dpi=set_dpi, format='jpg', bbox_inches='tight')
    #plt.show()
    