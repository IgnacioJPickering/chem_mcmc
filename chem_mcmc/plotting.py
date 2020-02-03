r"""Plotting utilities"""

def plot_minimization(minimization_position, lower, upper):
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  ax.plot(minimization_position, linewidth=0.5)
  ax.set_xlabel('Minimization step')
  ax.set_ylabel(r'Position ($\AA$)')
  ax.set_ylim(lower, upper)
  plt.show()

def plot_energy(tot_traj, mean_tot, heat_capacity=None):
  fig, ax = plt.subplots(1,1, figsize=(8,8))
  ax.plot(tot_traj, linewidth = 0.5, label='Total', color='g')
  ax.set_xlabel('MCMC steps')
  ax.set_ylabel(r'Energy ($kcal$ $mol^{-1}$)')
  xlo, xhi = ax.get_xlim()
  ax.hlines(mean_tot, xmin=xlo, xmax=xhi, 
           linestyles='dashed', colors='darkgreen', 
           label=f'Mean Total {mean_tot:.4f}'r' ($kcal$ $mol^{-1}$)')
  if heat_capacity is not None:
      ax.hlines([mean_tot+std_tot, mean_tot - std_tot], xmin=xlo, xmax=xhi,
              linestyles='dashed', colors='darkgreen',
              linewidths=[0.5, 0.5], 
              label=f'Heat capacity '
              '{heat_capacity:.4f}'r' ($kcal$ $mol^{-1}$ $K^{-1}$)')
  ax.legend(loc='upper right')
  ax.set_xlim(xlo, xhi)
  plt.show()

def plot_analysis(position, pdensity, lower, upper, title=None):
  fig, ax = plt.subplots(1,4, sharex=True,figsize=(18,6))
  dummy = np.linspace(lower, upper, 200)
  ax[0].scatter(position,range(len(position)), s=1.5)
  ax[0].set_ylabel('MCMC steps')
  ax[0].set_xlabel(r'Position ($\AA$)')
  ax[0].set_ylim(0, len(position))
  ax[1].plot(dummy, pdensity(dummy), )
  ax[1].set_ylabel('Probability density')
  ax[1].set_xlabel(r'Position ($\AA$)')
  ax[1].set_ylim(ymin=0)
  ax[2].hist(position, bins=80, facecolor='blue', alpha=0.5,
             density=True, orientation='vertical')
  ax[2].set_ylabel('Estimated frequency')
  ax[2].set_xlabel(r'Position ($\AA$)')
  ax[2].set_xlim(lower, upper)
  ax[3].plot(dummy, pdensity.potential(dummy))
  ax[3].set_ylabel(r'Potential ($kcal/mol$)')
  ax[3].set_xlabel(r'Position ($\AA$)')
  ax[2].set_xlim(lower, upper)
  ax[3].set_ylim(min(potential(dummy)), min(1.2, max(potential(position))))
  if title is not None:
    fig.suptitle(title)
  plt.show()
