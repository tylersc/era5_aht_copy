import xarray as xr
import numpy as np

import sys  
sys.path.insert(0, '../')
import ERA5_functions as era_fncts

weight = np.load('aht_weights.npy')

year = 2022

times = '18'

range1 = [31, 40, 50, 60, 70, 80]
range2 = [40, 50, 60, 70, 80, 90]

vcomp, temp, sphum, geo_pot = era_fncts.aht_opener_helper(year, times)

vcomp = vcomp.sel(expver=5)
temp = temp.sel(expver=5)
sphum = sphum.sel(expver=5)
geo_pot = geo_pot.sel(expver=5)

for i in range(len(range1)):
    for t in range(range1[i], range2[i]):
        new_ds = era_fncts.aht_instant(era_fncts.aht_time_sel_helper(vcomp, temp, sphum, geo_pot, t), weight)
        if t == range1[i]:
            full_ds = new_ds
        else:
            full_ds = xr.concat([full_ds, new_ds], 'time')

    #full_ds
    full_ds.to_netcdf('../aht_calcs/' + str(year) + '/' + str(year) + '_' + times + 'z_' + str(range1[i]) + '_' + str(range2[i]-1))
