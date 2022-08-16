import xarray as xr
from glob import glob
import numpy as np

lats = [-74, -75, -76]

for lat in lats:
    for year in range(1979, 2022, 1):
        which_year = str(year)
        print(which_year + str(lat))
        ddir = '../aht_calcs/' + which_year + '/'
        which_lat = lat

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

        eddy_all_times = sorted_ds.eddy_tot_int.sel(latitude=which_lat, method='nearest').sum(['level'], skipna=True).values
        #mmc_all_times = sorted_ds.mmc_tot_int.sel(latitude=which_lat, method='nearest').sum(['level'], skipna=True).values

        #np.save('../aht_time_series/one_lat/mmc_total_' + str(which_lat) + 'deg_all_times_' + which_year, mmc_all_times)
        np.save('../aht_time_series/one_lat/eddy_total_' + str(which_lat) + 'deg_all_times_' + which_year, eddy_all_times)