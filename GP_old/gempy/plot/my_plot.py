import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
def plot_a_block(block,receivers,num_rec = 0,section = None,direction = 'x'):
    _, ax = plt.subplots()
    block_ = block.reshape([receivers.n_devices, -1])[num_rec]
    block_mat = block_.reshape(receivers.kernel_resolution)
    extent = (0, block_mat.shape[1], block_mat.shape[0], 0)
    if direction == 'x':
        im = ax.imshow(block_mat[:,section,:].T,cmap = 'viridis',origin = 'lower',extent = extent)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    if direction == 'y':
        im = ax.imshow(block_mat[section,:,:].T,cmap = 'viridis',origin = 'lower',extent = extent)
        ax.set_xlabel('y')
        ax.set_ylabel('z')

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    grid_linewidth = 1
    ax.grid(which = 'minor',color='w', linestyle='-', linewidth=grid_linewidth)
    ax.grid(which = 'major',color='w', linestyle='-', linewidth=grid_linewidth)
    plt.colorbar(im)