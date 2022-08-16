import xarray as xr
import numpy as np

import sys  
sys.path.insert(0, '../')
import ERA5_functions as era_fncts

weight = np.load('aht_weights.npy')

year = 2021

times = '12'

range1 = np.asarray(range(0, 360, 10))
range2 = np.asarray(range(10, 370, 10))

range1 = np.append(range1, 360)

if year in range(1980, 2030, 4):
    range2 = np.append(range2, 366)
else:
    range2 = np.append(range2, 365)

vcomp, temp, sphum, geo_pot = era_fncts.aht_opener_helper(year, times)

for i in range(len(range1)):
    for t in range(range1[i], range2[i]):
        new_ds = era_fncts.aht_instant(era_fncts.aht_time_sel_helper(vcomp, temp, sphum, geo_pot, t), weight)
        if t == range1[i]:
            full_ds = new_ds
        else:
            full_ds = xr.concat([full_ds, new_ds], 'time')

    #full_ds
    full_ds.to_netcdf('../aht_calcs/' + str(year) + '/' + str(year) + '_' + times + 'z_' + str(range1[i]) + '_' + str(range2[i]-1))
