import xarray as xr
from glob import glob
import numpy as np

which_year = '1985'
ddir = 'aht_calcs/' + which_year + '/'

dfiles_00z = sorted(glob(ddir + which_year + '_00z*'))
#mfds_00z = xr.open_mfdataset(dfiles_00z, parallel=True)
mfds_00z = xr.open_mfdataset(dfiles_00z, concat_dim="time", combine="nested",
                             data_vars='minimal', coords='minimal', compat='override')

dfiles_06z = sorted(glob(ddir + which_year + '_06z*'))
#mfds_06z = xr.open_mfdataset(dfiles_06z, parallel=True)
mfds_06z = xr.open_mfdataset(dfiles_06z, concat_dim="time", combine="nested",
                             data_vars='minimal', coords='minimal', compat='override')

dfiles_12z = sorted(glob(ddir + which_year + '_12z*'))
#mfds_12z = xr.open_mfdataset(dfiles_12z)
mfds_12z = xr.open_mfdataset(dfiles_12z, concat_dim="time", combine="nested",
                             data_vars='minimal', coords='minimal', compat='override')

dfiles_18z = sorted(glob(ddir + which_year + '_18z*'))
#mfds_18z = xr.open_mfdataset(dfiles_18z)
mfds_18z = xr.open_mfdataset(dfiles_18z, concat_dim="time", combine="nested",
                             data_vars='minimal', coords='minimal', compat='override')
    
all_ds = xr.concat([mfds_00z, mfds_06z, mfds_12z, mfds_18z], dim='time')
    
sorted_ds = all_ds.sortby('time')

which_lev = 850
eddy_all_times = sorted_ds.eddy_tot_int.mean(['longitude'], skipna=True).sel(level=which_lev, method='nearest').values
mmc_all_times = sorted_ds.mmc_tot_int.sel(level=which_lev, method='nearest').values

np.save('aht_time_series/mmc_all_times_' + which_year + '_' + str(which_lev) +'hPa', mmc_all_times)
np.save('aht_time_series/eddy_all_times_' + which_year + '_' + str(which_lev) +'hPa', eddy_all_times)