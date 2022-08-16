''' Functions that should work for loading and using ERA5 data

load_era5 - Loads ERA5 data from the atmos server
time_spectra - Computes the power spectrum for a time series. Assumes 6 hourly data
space_time_spectrum - Performs space/time spectral analysis
time_filter - Uses a Butterworth filter to filter data
wavenumber_decomp - Perform a spatial fft to get the wavenumber breakdown
wavenumber_decomp2 - Computes the wavenumber decomp for a time series with python functions
get_lat_idx - Returns the index of given latitude
aht_weights - Gets the time invariant weighting for the AHT calculation
zonal_norm - Makes zonal norms that helps take weights in aht_instant
aht_opener_helper - Opens datasets that get fed into aht_time_sel_helper for a given year and time of day
aht_time_sel_helper - Takes datasets from aht_opener_helper, takes a time slice and returns numpy arrays
aht_instant - Calculate instantaneous AHT per Aaron's method
calc_aht_messori - Calculate AHT as in Messori/Czaja papers
calc_vert_ave - Get the vertically averaged parts of the AHT calculation for a given year
remove_seasons - Given a time series, removes the annual cycle assuming 6-hrly data
get_year_start_idx - Given an year, returns the index of the start of that year
get_times_of_idx - Given an index of the time series, returns datetime info about it
get_ndjf_data - Given a time series of data, returns only the ndjf parts of it
get_mjja_data - Given a time series of data, returns only the mjja parts of it
grab_omega_data - Opens the omega data corresponding to a given datetime
get_surface_level - Takes a dataset with level,lat,lon and returns the near-surface level
grab_mse_data - Opens the near-surface MSE data corresponding to before and after given datetime
grab_aht_data - Opens the AHT data corresponding to a given datetime
grab_temp_sphum_data - Creates a time-series of zonal-mean temp/sphum before and after a datetime
find_nearest - Given an array and a value, returns the element of the aray closest to that value and its index
convert_to_xarray - Works with time_selector to create xarray datasets of the data for a particular time
time_selector - Accepts ERA5 data and some info on the time of interest and returns numpy arrays of data for that time.
decorr_length_scales - Accepts AHT data at one latitude and returns info on zonal-decorrelation length scale.
plot_hist_and_gauss - Takes data and an axis and plots a histogram and Gaussian fit of the data
'''

import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.stats as sc
import copy
import metpy as mtp
from scipy import integrate
from scipy import interpolate
#0import xesmf as xe
import scipy.io as io
import pandas as pd
import datetime as dt
import math


#Constants
a=6371220 #radius of earth in m
L=2.5E6 #Latent heat in atmosphere
L_ice=3.34e5 #Latent heat of fusion
g=9.81 #Acceleration from gravity
conv_pw=1e15 #Conversion from watts to PW
cp=1007 
    
lats = np.linspace(90, -90, 361)
lons = np.linspace(0, 359.5, 720)
levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350,
            400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
geometry = 2 * np.pi * np.cos(np.deg2rad(np.asarray(lats))) * a / g
    

def load_era5(year):
    '''Given a year or the first part of a year it loads the data and turns it into one xarray dataset
    
    Args:
        -year (int)
        
    Returns:
        -dataset (xarray dataset)
    
    '''
    
    if type(year)!=int:
        print('Error with year type')
        
    else:
        pass
    
    year = str(year)
    
    ddir1 = '/home/disk/eos9/ERA5/hourly_pl/00/'
    dfiles1 = sorted(glob(ddir1 + year+'*.nc'))

    mfds1 = xr.open_mfdataset(dfiles1)

    ddir2 = '/home/disk/eos9/ERA5/hourly_pl/06/'
    dfiles2 = sorted(glob(ddir2 + year+'*.nc'))

    mfds2 = xr.open_mfdataset(dfiles2)

    ddir3 = '/home/disk/eos9/ERA5/hourly_pl/12/'
    dfiles3 = sorted(glob(ddir3 + year+'*.nc'))

    mfds3 = xr.open_mfdataset(dfiles3)

    ddir4 = '/home/disk/eos9/ERA5/hourly_pl/18/'
    dfiles4 = sorted(glob(ddir4 + year+'*.nc'))

    mfds4 = xr.open_mfdataset(dfiles4)
    
    all_ds = xr.concat([mfds1, mfds2, mfds3, mfds4], dim='time')
    
    sorted_ds = all_ds.sortby('time')
    
    return sorted_ds


def time_spectra(time_series, sig_level=0.95):
    '''Computes the power spectrum for a time series. Assumes 6 hourly data
    
    Args:
        -time_series(array): an array of time series data
        -sig_level(float): The significance level for the significance test
    
    Returns:
        -f_x(array-like): The frequency space for the spectrum
        -Px(array-like): The power in units**2
        -rspec(array-like): The spectrum of the red noise
        -end_ratio(array-like): The ratio to multiply the red noise spectrum by for the specific data
        -F_crit(float): The F-statistic for the significance test
    '''
    
    #Remove any mean
    time_series = time_series - time_series.mean()
    
    nperseg=None
    wintype='boxcar'
    scaling='spectrum'
    detrend=False
    
    #f_x, Px = signal.welch(time_series, fs=4, nperseg=chunk_length, scaling=which_scaling, detrend=False, noverlap=0, average='mean')
    f_x, Px = signal.csd(time_series, time_series, fs=4, window=wintype, return_onesided=True, scaling=scaling, nperseg=nperseg,  detrend=detrend, average='mean')
    
    #make red noise
    #finding autocorrelation values for alpha values

    #This was just from Dennis's code on SST stuff
    #I think some of the means are unneccessary as it is just a single value, but oh well
    total_length=len(time_series)       
    vv=np.matmul(time_series,np.transpose(time_series))/total_length
    one_lag_cov=np.matmul(time_series[:-1], np.transpose(time_series[1:]))/(total_length-1)
    ratio=one_lag_cov/vv
    autt=np.mean(ratio*vv)/np.mean(vv)

    #make some red spectra

    chunk_length = 256 #default I think?
    n=int(chunk_length/2)
    alpha=autt

    rspec=np.empty(n)

    for i in range(0,n):
        rspec[i]=(1-alpha**2)/(1-2*alpha*np.cos(i*np.pi/n)+alpha**2)

    #now we will need to scale the red noise by summing the powers and multiplying by the ratios

    total_int=np.sum(Px)
    total_red_noise=np.sum(rspec)
    end_ratio=total_int/total_red_noise

    #doing a significance test

    #ddof is the degrees of freedom for each dataset

    N=total_length
    fw=1.2 #fudge factor for overlapping windows
    M=chunk_length

    ddof=2*fw*N/M
    ddof_red=(N-2)/2

    #F_stats=Px[:-1]/(rspec_1*ratio1)  #Not sure what this is for
    F_crit=sc.f.ppf(q=sig_level, dfn=ddof, dfd=ddof_red)
    
    return [f_x, Px, rspec, end_ratio, F_crit]


def space_time_spectrum(datas, kmax=20, chunk_length=256):
    '''Gotten from Andrew Pauling 6-15-21
    Adapted by Tyler Cox
    Performs space/time spectral analysis
    
    Args:
        -datas(array): Needs to have axis of time, longitude
        -kmax(int): Maximum spatial wavenumber. Defaults to 20
        -chunk_length(int): Chunk length for the window size, defaults to 256
        
    Output:
        -P(array): Computers space-time power spectrum
    '''

    #datas = np.asarray(datas)
    
    # Get shape of input array
    timx, lonx = datas.shape

    # nfft = 64 # number of fft points
    chunk = chunk_length  # length of chunk
    N_chunks = int(2*np.floor(timx/chunk))  # Number of chunks
    noverlap = chunk/2  # 50% overlap

    # Do spatial fft
    Fs = np.fft.fft(datas, axis=1, norm='ortho')

    # Get Fourier coefficients following Hartmann and Barnes notes
    C = 2*np.real(Fs)
    K = -2*np.imag(Fs)

    # kmax = 20 # Maximum spatial wavenumber
    IMAX = int(chunk/2)  # time axis
    JMAX = int(np.floor(lonx/2))  # space axis

    Pmat = np.zeros((N_chunks, IMAX, 2*kmax+1))

    # Create Hamming window
    window = signal.windows.hamming(chunk_length)
    window = np.tile(window[:, None], (1, lonx))

    for n in range(N_chunks):   # Loop over chunks
        ind1 = int(n*noverlap)  # Starting index
        ind2 = int(ind1+chunk)  # End index

        # Get chunks
        C_chunk = window*C[ind1:ind2, :]
        K_chunk = window*K[ind1:ind2, :]

        # Do FFT on each part
        Ftc = np.fft.fft(C_chunk, axis=0, norm='ortho')
        Ftk = np.fft.fft(K_chunk, axis=0, norm='ortho')

        # Get Fourier coefficients
        A = 2*np.real(Ftc)
        B = -2*np.imag(Ftc)

        a = 2*np.real(Ftk)
        b = -2*np.imag(Ftk)

        # Set up east and west propagating matrices
        stw = np.zeros([IMAX, JMAX])
        ste = np.zeros([IMAX, JMAX])

        # Set up full matrix
        st = np.zeros([IMAX, 2*kmax+1])

        # Compute power spectrum following Hayashi (1971)
        for i in range(IMAX):
            for j in range(JMAX):
                stw[i, j] = 1/8*((A[i, j]-b[i, j])**2 + (-B[i, j]-a[i, j])**2)
                ste[i, j] = 1/8*((A[i, j]+b[i, j])**2 + (B[i, j]-a[i, j])**2)

        for i in range(IMAX):
            st[i, kmax] = ste[i, 0]
            for k in range(kmax):
                st[i, kmax+(k+1)] = ste[i, k+1]
                st[i, kmax-(k+1)] = stw[i, k+1]

        # Put chunk into power spectrum matrix
        Pmat[n, :, :] = st

    # Average over chunks
    P = np.mean(Pmat, axis=0)

    return P


def time_filter(time_series, filter_type, one_cut, other_cut=None, chunk_length=256):
    '''
    Uses a Butterworth filter to filter data
    Args:
        -time_series(array): an array of time series data
        -filter_type(str): The type of filter ('low', 'high' or 'band')
        -one_cut(float): Frequency of to cut for low or highpass filter, or lower cutoff for bandpass
        -other_cut(float): Upper frequency for bandpass filter
        -chunk_length(int): Chunk length for the window size, defaults to 256
    
    Returns:
        -f_x(array-like): The frequency space for the spectrum
        -Px(array-like): The power in units**2
    
    '''
    
    order=6 #for butterworth filter

    #Remove any mean
    time_series = time_series - time_series.mean()
    
    if filter_type == 'low':
        sos_filter = signal.butter(order, one_cut, btype='lowpass', fs=4, output='sos')
        
    elif filter_type == 'high':
        sos_filter = signal.butter(order, one_cut,  btype='highpass', fs=4, output='sos')
    
    elif filter_type == 'band':
        sos_filter = signal.butter(order, [one_cut, other_cut], btype='band', fs=4, output='sos')
        
    else:
        print('Band type error')

    filtered_data=signal.sosfilt(sos_filter, time_series)

    #f_1_filtered, P1_filtered = signal.welch(filtered_data, fs=4, nperseg=chunk_length, scaling='spectrum', average='median')
    
    nperseg=None
    wintype='boxcar'
    scaling='spectrum'
    detrend=False
    
    #f_x, Px = signal.welch(time_series, fs=4, nperseg=chunk_length, scaling=which_scaling, detrend=False, noverlap=0, average='mean')
    f_filtered, P_filtered = signal.csd(filtered_data, filtered_data, fs=4, window=wintype, return_onesided=True, scaling=scaling, nperseg=nperseg,  detrend=detrend, average='mean')

    return(f_filtered, P_filtered)



def wavenumber_decomp(signal):
    '''Perform a spatial fft to get the wavenumber breakdown
    NEED TO ADD SOME WINDOWING TO THIS I THINK
    
    Args:
        -signal(array): an array of the spatial data - must be 1D currently
    
    Returns:
        -Fs(array-like): The power in units**2
        -frequency_space(array-like): The wavenumber space for the spectrum
        -phase(array-like): Phase for each wavenumber
    
    '''
    
    #Total power is np.sum(Fs) / len(signal) ** 2
    signal = signal-signal.mean()
    Fs_raw = np.fft.fft(signal)
    
    #Take only first half (positive frequencies) of data
    #So  need to square it to get variance and multiply by 2
    
    Fs = np.abs(Fs_raw[:len(signal)//2]) ** 2 * 2
    
    frequency_space = np.fft.fftfreq(signal.size, d=1/720)[:len(signal)//2]
    
    phase = np.rad2deg(np.angle(Fs_raw))[:len(signal)//2]
    
    return frequency_space, Fs, phase


def wavenumber_decomp2(datas):
    '''Computes the power spectrum for a time series. Assumes 6 hourly data
    
    Args:
        -time_series(array): an array of time series data
        -sig_level(float): The significance level for the significance test
    
    Returns:
        -f_x(array-like): The frequency space for the spectrum
        -Px(array-like): The power in units**2
        -rspec(array-like): The spectrum of the red noise
        -end_ratio(array-like): The ratio to multiply the red noise spectrum by for the specific data
        -F_crit(float): The F-statistic for the significance test
    '''
    
    #Remove any mean
    datas = datas - datas.mean()
    
    nperseg=None
    wintype='boxcar'
    scaling='spectrum'
    detrend=False
    
    f_x, Px = signal.welch(datas, window=wintype, return_onesided=False, scaling=scaling, nperseg=nperseg,  detrend=detrend, average='mean')
    #f_x, Px = signal.csd(datas, datas, window=wintype, return_onesided=False, scaling=scaling, nperseg=nperseg,  detrend=detrend, average='mean')
    
    #make red noise
    #finding autocorrelation values for alpha values

    #This was just from Dennis's code on SST stuff
    #I think some of the means are unneccessary as it is just a single value, but oh well
    #total_length=len(time_series)       
    #vv=np.matmul(time_series,np.transpose(time_series))/total_length
    #one_lag_cov=np.matmul(time_series[:-1], np.transpose(time_series[1:]))/(total_length-1)
    #ratio=one_lag_cov/vv
    #autt=np.mean(ratio*vv)/np.mean(vv)

    #make some red spectra

    #n=int(chunk_length/2)
    #alpha=autt

    #rspec=np.empty(n)

    #for i in range(0,n):
        #rspec[i]=(1-alpha**2)/(1-2*alpha*np.cos(i*np.pi/n)+alpha**2)

    #now we will need to scale the red noise by summing the powers and multiplying by the ratios

    #total_int=np.sum(Px)
    #total_red_noise=np.sum(rspec)
    #end_ratio=total_int/total_red_noise

    #doing a significance test

    #ddof is the degrees of freedom for each dataset

    #N=total_length
    #fw=1.2 #fudge factor for overlapping windows
    #M=chunk_length

    #ddof=2*fw*N/M
    #ddof_red=(N-2)/2

    #F_stats=Px[:-1]/(rspec_1*ratio1)  #Not sure what this is for
    #F_crit=sc.f.ppf(q=sig_level, dfn=ddof, dfd=ddof_red)
    
    return f_x, Px


def get_lat_idx(which_lat):
    '''Returns the index of given latitude
    
    Args:
        -which_lat(int): Latitude that you want the index for
        
    Returns:
        -lat_idx(int): Index of the latitude
    '''
     
    lats = np.linspace(90, -90, 361)
    lat_idx = (np.abs(lats - which_lat)).argmin()
    
    return lat_idx

def get_lon_idx(which_lon):
    '''Returns the index of given latitude
    
    Args:
        -which_lat(int): Latitude that you want the index for
        
    Returns:
        -lat_idx(int): Index of the latitude
    '''
     
    lons = np.linspace(0, 359.5, 720)
    lon_idx = (np.abs(lons - which_lon)).argmin()
    
    return lon_idx


def aht_weights():
    '''Gets the time invariant weighting for the AHT calculation
    
    Agrs:
        None
    
    Returns:
        -weight (array-like): Time-invariant array of the vertical weighting for AHT (coords of level, lat, lon)
    
    '''
    
    #load in the surface pressure
    ps = xr.open_dataset('aht_calcs/monthly_ps.nc')
    #aarons ps
    #aarons_ps = io.loadmat('aaron_aht_stuff/CLIM_PS.mat', verify_compressed_data_integrity=False)
    #ps = aarons_ps['CPS']
    
    #Load in the era5 data to get the proper lat/lon and levels
    dummy_era5_data = xr.open_dataset('/home/disk/eos9/ERA5/hourly_pl/00/1985.v.nc')
    
    lat = dummy_era5_data.latitude
    lon = dummy_era5_data.longitude
    
    new_coords = xr.Dataset({'lat': (['lat'], dummy_era5_data.latitude.values),
                         'lon': (['lon'], dummy_era5_data.longitude.values),
                        }
                       )

    regridder = xe.Regridder(ps, new_coords, 'bilinear')

    # the entire dataset can be processed at once
    new_ps = regridder(ps)
    
    ps=new_ps.sp.mean(['time']).sel(expver=1) #Surface pressure

    

    #Getting our initial half pressure levels
    pres_halfs = copy.deepcopy(dummy_era5_data.level.values)

    pres_halfs_dif = (np.asarray(pres_halfs[1:]) + np.asarray(pres_halfs[:-1])) / 2
    final_pres_halfs = xr.DataArray(np.append(np.insert(pres_halfs_dif,0,0), 1013))

    #pres_half_new_bot= np.expand_dims(final_pres_halfs, axis=(1,2))#.astype(np.float32)
    pres_half_new_bot=copy.deepcopy(final_pres_halfs.
                                        expand_dims({'lat':len(lat)},1).expand_dims({'lon':len(lon)},-1).values)

    #Now turn any levels below surface pressure to Nans
    for k in range(0,len(final_pres_halfs),1):
        pres_half_new_bot[k,:,:][pres_half_new_bot[k,:,:] > ps.values/100]=np.nan
        #pres_half_new_bot[k,:,:][pres_half_new_bot[k,:,:] > ps/100]=np.nan

    #Now get the index of the lowest non-nan level so we can turn it to surface pressure
    low_level_index=(~np.isnan(pres_half_new_bot)).cumsum(0).argmax(0)

    #Make it so that if not the lowest level, the level is added to where the first nan is, not the first non-nan
    low_level_index[low_level_index<37] += 1
    #Now add the surface pressure
    np.put_along_axis(pres_half_new_bot, low_level_index[None,:,:], (ps.values/100)[None,:,:], axis=0)
    #np.put_along_axis(pres_half_new_bot, low_level_index[None,:,:], (ps/100)[None,:,:], axis=0)

    #Now finite difference it to get weight
    weight=np.diff(100*pres_half_new_bot, axis=0)

    weight[weight<0]=np.nan   #Trying a nan here instead of 0
    
    return weight

def zonal_norm(weight):
    '''Makes zonal norms that helps take weights in aht_instant
    Args:
        -weight(array): Array of vertical weights from aht_weights
        
    Returns:
        -zon_norm(array): Zonal norms to be used in aht_instant to take zonal averages
    '''
    
    zonal_sum = np.nansum(weight, axis=2)

    zon_norm = weight/zonal_sum[:,:,None]

    zon_norm[np.isnan(zon_norm)] = 0
    
    return zon_norm


def aht_opener_helper(year, hour):
    '''Opens datasets that get fed into aht_time_sel_helper for a given year and time of day
    Args:
        -year(int): Year of data to get
        -hour(str): Time of day of data to get
        
    Returns:
        -vcomp(dataset): Dataset of meridional wind for the year
        -temp(dataset): Dataset of temperature for the year
        -sphum(dataset): Dataset of specific humidity for the year
        -geo_pot(dataset): Dataset of geopotential for the year
    '''
    
    vcomp = xr.open_dataset('/tdat/tylersc/era5_aht/era5_raw_data/' + str(hour) + '/' + str(year) + '.v_component_of_wind.nc')
    temp = xr.open_dataset('/tdat/tylersc/era5_aht/era5_raw_data/' + str(hour) + '/' + str(year) + '.temperature.nc')
    sphum = xr.open_dataset('/tdat/tylersc/era5_aht/era5_raw_data/' + str(hour) + '/' + str(year) + '.specific_humidity.nc')
    geo_pot = xr.open_dataset('/tdat/tylersc/era5_aht/era5_raw_data/' + str(hour) + '/' + str(year) + '.geopotential.nc')
    
    return vcomp, temp, sphum, geo_pot
    
def aht_time_sel_helper(vcomp, temp, sphum, geo_pot, time_idx):
    '''Takes datasets from aht_opener_helper, takes a time slice and returns numpy arrays
    Args:
        -vcomp(dataset): Year of meridional wind data
        -temp(dataset): Year of temperature data
        -sphum(dataset): Year of specific humidity data
        -geo_pot(dataset): Year of geopotential data
        -time_idx(int): Time index for the instant to calculate
        
    Returns:
        -vcomp_np(array): Numpy array of meridional wind for the time index
        -temp_np(array): Numpy array of temperature for the time index
        -sphum_np(array): Numpy array of specific humidity for the time index
        -geo_pot_np(array): Numpy array of geopotential for the time index
        -time_point(datetime): Datetime object of the time index
    '''
    
    
    vcomp_np = (vcomp.v.isel(time=time_idx)).values
    temp_np = (temp.t.isel(time=time_idx)).values
    sphum_np = (sphum.q.isel(time=time_idx)).values
    geo_pot_np = (geo_pot.z.isel(time=time_idx)).values
    
    time_point = vcomp.time.isel(time=time_idx)
    
    return vcomp_np, temp_np, sphum_np, geo_pot_np, time_point



def aht_instant(datas_np, weight):
    '''Calculate instantaneous AHT per Aaron's method
    Args:
        -datas_np(array): Meridional wind, temp, sphum, and geopotential in numpy forms
        -weight(array): Vertical weights to use from aht_weights function
        
    Returns:
        -final_ds(dataset): Dataset of AHT for one instant
    '''         
    
    temp_mse_units = (datas_np[1] * cp)
    sphum_mse_units = (datas_np[2] * L)
    z_mse_units = (datas_np[3])
    mse = temp_mse_units + sphum_mse_units + z_mse_units
    vcomp = (datas_np[0])

    time_point = datas_np[4]  #The time of the data
    
    #MMC
    
    zon_norms = np.load('zonal_norms.npy')
    
    weight[np.isnan(weight)] = 0

    weight_zonal_ave = np.nanmean(weight, axis=2)
    
    #Now make sure we don't count nan levels in zonal mean
    nan_count=np.isnan(weight).sum(axis=2)
    nan_count=1-(nan_count/len(weight[0,0,:]))
    weight_zonal_ave_new=weight_zonal_ave*nan_count
    
    temp_zon_mean = np.nansum(temp_mse_units * zon_norms, axis=2)
    sphum_zon_mean = np.nansum(sphum_mse_units * zon_norms, axis=2)
    geo_pot_zon_mean = np.nansum(z_mse_units * zon_norms, axis=2)
    mse_zon_mean = np.nansum(mse * zon_norms, axis=2)
    vcomp_zon_mean = np.nansum(vcomp * zon_norms, axis=2)
    
    temp_vert_ave = np.nanmean(temp_zon_mean * weight_zonal_ave, axis=0) / np.nanmean(weight_zonal_ave, axis=0)
    sphum_vert_ave = np.nanmean(sphum_zon_mean * weight_zonal_ave, axis=0) / np.nanmean(weight_zonal_ave, axis=0)
    z_vert_ave = np.nanmean(geo_pot_zon_mean * weight_zonal_ave, axis=0) / np.nanmean(weight_zonal_ave, axis=0)
    mse_vert_ave = np.nanmean(mse_zon_mean * weight_zonal_ave, axis=0) / np.nanmean(weight_zonal_ave, axis=0)
    vcomp_vert_ave = np.nanmean(vcomp_zon_mean * weight_zonal_ave, axis=0) / np.nanmean(weight_zonal_ave, axis=0)

    temp_mmc = temp_zon_mean - np.expand_dims(temp_vert_ave, 0)
    sphum_mmc = sphum_zon_mean - np.expand_dims(sphum_vert_ave, 0)
    z_mmc = geo_pot_zon_mean - np.expand_dims(z_vert_ave, 0)
    mse_mmc = mse_zon_mean - np.expand_dims(mse_vert_ave, 0)
    vcomp_mmc = vcomp_zon_mean - np.expand_dims(vcomp_vert_ave, 0)
    
    moc_tot_int = mse_mmc * vcomp_mmc * weight_zonal_ave
    moc_temp_int = temp_mmc * vcomp_mmc * weight_zonal_ave
    moc_geo_pot_int = z_mmc * vcomp_mmc * weight_zonal_ave
    moc_dry_int = moc_temp_int + moc_geo_pot_int
    moc_moist_int = sphum_mmc * vcomp_mmc * weight_zonal_ave

    #moc_tot = np.nansum(mse_mmc * vcomp_mmc * weight_zonal_ave, axis=0)
    #moc_temp = np.nansum(temp_mmc * vcomp_mmc * weight_zonal_ave, axis=0)
    #moc_geo_pot = np.nansum(z_mmc * vcomp_mmc * weight_zonal_ave, axis=0)
    #moc_dry = moc_temp + moc_geo_pot
    #moc_moist = np.nansum(sphum_mmc * vcomp_mmc * weight_zonal_ave, axis=0)

    #Eddy

    temp_zonal_anom = temp_mse_units - np.expand_dims(np.nanmean(temp_mse_units, axis=2), 2)
    sphum_zonal_anom = sphum_mse_units - np.expand_dims(np.nanmean(sphum_mse_units, axis=2), 2)
    z_zonal_anom = z_mse_units - np.expand_dims(np.nanmean(z_mse_units, axis=2), 2)
    mse_zonal_anom = mse - np.expand_dims(np.nanmean(mse, axis=2), 2)
    vcomp_zonal_anom = vcomp - np.expand_dims(np.nanmean(vcomp, axis=2), 2)

    eddy_tot_int = mse_zonal_anom * vcomp_zonal_anom * weight
    eddy_temp_int = temp_zonal_anom * vcomp_zonal_anom * weight
    eddy_geo_pot_int = z_zonal_anom * vcomp_zonal_anom * weight
    eddy_dry_int = eddy_temp_int + eddy_geo_pot_int
    eddy_moist_int = sphum_zonal_anom * vcomp_zonal_anom * weight
    
    #eddy_tot = np.nansum(eddy_tot_int, axis=0)
    #eddy_temp = np.nansum(eddy_temp_int, axis=0)
    #eddy_geo_pot = np.nansum(eddy_geo_pot_int, axis=0)
    #eddy_dry = eddy_temp + eddy_geo_pot
    #eddy_moist = np.nansum(eddy_moist_int, axis=0)
    
    #Make xarray dataset to return
    
    lats = np.linspace(90, -90, 361)
    lons = np.linspace(0, 359.5, 720)
    levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
    

    final_ds = xr.Dataset(
                        data_vars = dict(
                            #mmc_tot=(['latitude', 'longitude'], moc_tot),
                            #mmc_dry=(['latitude', 'longitude'], moc_dry),
                            #mmc_moist=(['latitude', 'longitude'], moc_moist),
                            #eddy_tot=(['latitude', 'longitude'], eddy_tot),
                            #eddy_dry=(['latitude', 'longitude'], eddy_dry),
                            #eddy_moist=(['latitude', 'longitude'], eddy_moist),
                            mmc_tot_int=(['level', 'latitude'], moc_tot_int),
                            mmc_dry_int=(['level', 'latitude'], moc_dry_int),
                            mmc_moist_int=(['level', 'latitude'], moc_moist_int),
                            eddy_tot_int=(['level', 'latitude', 'longitude'], eddy_tot_int),
                            eddy_dry_int=(['level', 'latitude', 'longitude'], eddy_dry_int),
                            eddy_moist_int=(['level', 'latitude', 'longitude'], eddy_moist_int),
                        ),
                        coords=dict(
                            time=time_point,
                            latitude=lats,
                            longitude=lons,
                            level=levels)
    )
    
    return final_ds
    
    
def calc_aht_messori(year):
    '''Calculate AHT as in Messori/Czaja papers
    Args:
        -year(int): Year to calculate
        
    Returns:
        -v_te(array): Time anomaly meridional wind
        -T_te(array): Time anomaly temperature
        -q_te(array): Time anomaly specific humidity
        -z_te(array): Time anomaly geopotential
        -v_monthly_ave(array): Monthyl mean of meridional wind
        -T_monthly_ave(array): Monthyl mean of temperature
        -q_monthly_ave(array): Monthyl mean of specific humiditiy
        -z_monthly_ave(array): Monthyl mean of geopotential
    '''
    
    ds = load_era5(1985).sel(level=850, method='nearest')

    # Calculate the weighted average
    ds_monthly_ave = (ds).groupby('time.month').mean(dim='time')
    
    ds_te = ds.groupby('time.month') - ds_monthly_ave
    
    v_te = ds_te.v
    T_te = ds_te.t
    q_te = ds_te.q
    z_te = ds_te.z
    
    v_monthly_ave = ds_monthly_ave.v
    T_monthly_ave = ds_monthly_ave.t
    q_monthly_ave = ds_monthly_ave.q
    z_monthly_ave = ds_monthly_ave.z
    
    return [v_te, T_te, q_te, z_te, v_monthly_ave, T_monthly_ave, q_monthly_ave, z_monthly_ave]

def calc_vert_ave(year):
    '''Get the vertically averaged parts of the AHT calculation for a given year
    Args:
        -year(int): Year to calculate
        
    Returns:
        -vcomp_no_vert_ave(array): Lat/lon of vertically averaged vcomp
        -temp_no_vert_ave(array): Lat/lon of vertically averaged temp
        -sphum_no_vert_ave(array): Lat/lon of vertically averaged sphum
        -geo_pot_no_vert_ave(array): Lat/lon of vertically averaged geo-potential
    '''
    
    ds = load_era5(1985)
    
    zon_norms = np.load('aht_calcs/zonal_norms.npy')
    weight = np.load('aht_calcs/aht_weights.npy')
    
    weight[np.isnan(weight)] = 0

    weight_zonal_ave = np.nanmean(weight, axis=2)
    
    # Calculate the weighted average
    ds_monthly_ave = (ds).groupby('time.month').mean(dim='time')
    
    vcomp = ds_monthly_ave.v
    temp = ds_monthly_ave.t * cp
    sphum = ds_monthly_ave.q * L
    geo_pot = ds_monthly_ave.z
    
    vcomp_vert_ave = np.nanmean(vcomp * weight[None,:,:,:], axis=1) / np.nanmean(weight, axis=0)[None,:,:]
    temp_vert_ave = np.nanmean(temp * weight[None,:,:,:], axis=1) / np.nanmean(weight, axis=0)[None,:,:]
    sphum_vert_ave = np.nanmean(sphum * weight[None,:,:,:], axis=1) / np.nanmean(weight, axis=0)[None,:,:]
    geo_pot_vert_ave = np.nanmean(geo_pot * weight[None,:,:,:], axis=1) / np.nanmean(weight, axis=0)[None,:,:]

    vcomp_no_vert_ave = vcomp - np.expand_dims(vcomp_vert_ave, 1)
    temp_no_vert_ave = temp - np.expand_dims(temp_vert_ave, 1)
    sphum_no_vert_ave = sphum - np.expand_dims(sphum_vert_ave, 1)
    geo_pot_no_vert_ave = geo_pot - np.expand_dims(geo_pot_vert_ave, 1)
    
    return [vcomp_no_vert_ave, temp_no_vert_ave, sphum_no_vert_ave, geo_pot_no_vert_ave]

def make_A_matrix(datas, harmonics = 5):
    '''Given a time series, removes the annual cycle assuming 6-hrly data
    Args:
        -harmonics(int): Number of annual harmonics to remove
        
    Returns:
        -A_matrix(array): Numpy array with specified number of harmonics
    '''
    
    #Let's get the first n periodicities

    n_periods = harmonics * 2 + 1 #This gives us the first #harmonics sin and cos

    #make a time space
    time_space=np.linspace(0, len(datas), len(datas))


    A_matrix=np.zeros((n_periods, len(datas)))

    f1=np.ones(len(datas))

    A_matrix[0,:]=f1

    counter = 1 
    for i in range(1,n_periods,2):
        A_matrix[i, :] = np.cos(counter*np.pi*time_space/365.25/4)
        counter += 1

    counter = 1
    for i in range(2,n_periods,2):
        A_matrix[i, :] = np.sin(counter*np.pi*time_space/365.25/4)
        counter += 1
        
    return A_matrix


def remove_seasons(datas, A_matrix, one_year=False):
    '''Given a time series, removes the annual cycle assuming 6-hrly data
    Args:
        -datas(array): Timeseries to remove annual cycle from
        -harmonics(int): Number of annual harmonics to remove
        
    Returns:
        -datas_no_season(array): Timeseries without annual cycle
        -x_squiggle(array): Annual cycle portion of the timeseries
    '''

    #do A*x_transpose/N
    
    #Remove mean
    datas_no_mean = datas - np.mean(datas)
    
    #detrend it
    #datas_no_trend_no_mean = signal.detrend(datas)
    #slope, intercept, r_value, p_value, std_err = sc.linregress(time_space, datas)
    #datas_no_trend_no_mean = datas - (time_space * slope + intercept)

    #A_xT=np.matmul(A_matrix,np.transpose(datas_no_trend_no_mean))/len(datas_no_trend_no_mean)
    A_xT=np.matmul(A_matrix,np.transpose(datas_no_mean))/len(datas_no_mean)

    #now find x_squiggle to remove from x
    x_squiggle = np.matmul(np.transpose(A_xT),A_matrix)

    #datas_no_season = datas_no_trend_no_mean - x_squiggle
    
    #Need the times two I belive because I double the number of harmonics as of 9/27/21
    #BUT this struggles to detrend one year of data
    #So going to add an if statement for it
    
    #datas_no_season = datas - x_squiggle
    
    if one_year == True:
        datas_no_season = datas_no_mean - x_squiggle
        
    else:
        datas_no_season = datas_no_mean - 2 * x_squiggle
    
    
    return datas_no_season, x_squiggle


def remove_seasons_spline(datas, num_knots=8, periodic=True):
    '''Given a time series, removes the annual cycle assuming 6-hrly data
    Args:
        -datas(array): Timeseries to remove annual cycle from
        -num_knots(int): Number of breakpoints (not including ends)
        
    Returns:
        -datas_no_season(array): Timeseries without annual cycle
        -seasonal_cycle(array): Annual cycle portion of the timeseries
    '''
    
    knots = np.linspace(0.25, (len(datas)/4) - .25, int((num_knots + 2) * len(datas)/4/365))

    time_space = np.linspace(0, len(datas)/4, len(datas))
    
    spl = interpolate.LSQUnivariateSpline(time_space, datas, knots)

    seasonal_cycle = spl(time_space)
    
    seasonal_ave = np.zeros(int(365.25*4))
    years_of_data = int(len(datas)/4/365.25)
    
    for i in range(years_of_data):
        seasonal_ave += seasonal_cycle[int(i*365.25*4):int((i+1)*365.25*4)]

    seasonal_cycle_periodic = np.tile(seasonal_ave/years_of_data, years_of_data)

    if len(datas) != len(seasonal_cycle_periodic): #We need to make these the same length in case leap year made it weird
        periodic_dummy_time = np.linspace(0, len(datas), len(seasonal_cycle_periodic))
        datas_dummy_time = np.linspace(0, len(datas), len(datas))

        interped_periodic_cycle = np.interp(datas_dummy_time, periodic_dummy_time, seasonal_cycle_periodic)
        seasonal_cycle_periodic = interped_periodic_cycle
    else:
        pass
    
    if periodic==True:
        datas_no_season = datas - seasonal_cycle_periodic
        seasonal_cycle = seasonal_cycle_periodic
        
    else:
        datas_no_season = datas - seasonal_cycle
    
    
    return datas_no_season, seasonal_cycle


def get_year_start_idx(year):
    '''Given an year, returns the index of the start of that year
    Args:
        -year(int): Year to get start of
        
    Returns:
        -year_idx(int): Index of the start of the year
    '''
        
    time_range = pd.date_range('1979-01-01', '2018-12-31 18:00:00', freq='6H')
    year_idx = time_range.get_loc(str(year) + '-01-01 00:00:00')
    
    return year_idx

def get_times_of_idx(idx):
    '''Given an index of the time series, returns datetime info about it
    Args:
        -idx(int): Time index
        
    Returns:
        -which_date(datetime date): Datetime date of the time index
        -time(datetime time): Datetime time of the time index
        -date_time(datetime): Datetime date/time of the time index
    '''
    
    #time_range = pd.date_range('1979-01-01', '2018-12-31 18:00:00', freq='6H')
    time_range = pd.date_range('1979-01-01', '2022-03-31 18:00:00', freq='6H')
    
    which_date = time_range[idx].date()
    time = time_range[idx].time()
    
    date_time = time_range[idx]
    
    return which_date, time, date_time, idx

def check_idx_months(date_time, idx, months=[11, 12, 1, 2]):
    '''Given an index of the time series, returns datetime info about it
    Args:
        -date_time(datetime like object): Datetime of event
        
    Returns:
        -idx(int): Index of value that is out of months range
    '''
    
    month=date_time.month
    
    if month not in months:
        #print('IDX NOT IN REQD MONTHS')
        #print('DATE IS: ' + str(date_time))
        #print('Index is: ' + str(idx))
        
        return idx
    else:
        return None
    
def get_ndjf_data(input_array):
    '''Given an array of timeseries AHT data, returns only the NDJF parts of the time series
    '''
    
    time_range = pd.date_range('1979-01-01', '2018-12-31 18:00:00', freq='6H')
    #time_range = pd.date_range('1980-01-01', '2018-12-31 18:00:00', freq='6H')
    
    valid_months = [11, 12, 1, 2]
    
    output_list = []
    
    for i in range(len(input_array)):
        which_month = time_range[i].month
        if which_month in valid_months:
            output_list.append(input_array[i])
            
    return np.asarray(output_list)

def get_djf_data(input_array):
    '''Given an array of timeseries AHT data, returns only the DJF parts of the time series
    '''
    
    time_range = pd.date_range('1979-01-01', '2018-12-31 18:00:00', freq='6H')
    #time_range = pd.date_range('1980-01-01', '2018-12-31 18:00:00', freq='6H')
    
    valid_months = [12, 1, 2]
    
    output_list = []
    
    for i in range(len(input_array)):
        which_month = time_range[i].month
        if which_month in valid_months:
            output_list.append(input_array[i])
            
    return np.asarray(output_list)

def get_mjja_data(input_array):
    '''Given an array of timeseries AHT data, returns only the DJF parts of the time series
    '''
    
    #time_range = pd.date_range('1979-01-01', '2018-12-31 18:00:00', freq='6H')
    time_range = pd.date_range('1980-01-01', '2018-12-31 18:00:00', freq='6H')
    
    valid_months = [5, 6, 7, 8]
    
    output_list = []
    
    for i in range(len(input_array)):
        which_month = time_range[i].month
        if which_month in valid_months:
            output_list.append(input_array[i])
            
    return np.asarray(output_list)


def grab_era5_data(datetime_info, field):
    '''Opens the omega data corresponding to a given datetime
    Args:
        -datetime_info(list): List of datetime info from get_times_of_idx function
        
    Returns:
        -ds_time_sel(dataset): Xarray dataset of the omega data for the datetime
    '''
    
    date = datetime_info[0]
    time = datetime_info[1]
    date_time = datetime_info[2]
    year = str(date)[:4]
    day_of_year = date.timetuple().tm_yday - 1 #So that the year starts at day 0
    time_of_day = str(time)[:2] #get only the 00z, 06z etc. part
    
    file_str = '/tdat/tylersc/era5_aht/era5_raw_data/' + time_of_day + '/' + year + '.' + field  + '.nc'
    
    ds = xr.open_dataset(file_str)

    ds_time_sel = ds.sel(time=date_time)
    
    return ds_time_sel


def get_surface_level(ds):
    '''Takes a dataset with level,lat,lon and returns the near-surface level
    Args:
        -ds(dataset): Original dataset
        
    Returns:
        -low_level_ds(dataset): Xarray dataset at the near-surface level
    '''

    zon_norms = np.load('aht_calcs/zonal_norms.npy')
    zon_norms[zon_norms==0] = np.nan

    low_level_index= (~np.isnan(zon_norms)).cumsum(0).argmax(0)

    index_da = xr.DataArray(low_level_index, dims=['latitude', 'longitude'])

    low_level_ds = ds.isel(level=index_da)
    
    return low_level_ds


def grab_mse_data(datetime_info, remove_season=True):
    '''Opens the near-surface MSE data corresponding to before and after given datetime
    Args:
        -datetime_info(list): List of datetime info from get_times_of_idx function
        
    Returns:
        -ds_before_level(dataset): Xarray dataset of the near-surface MSE for the two days before
        -ds_after_level(dataset): Xarray dataset of the near-surface MSE for the two days after
    
    '''
    
    date = datetime_info[0]
    time = datetime_info[1]
    date_time = datetime_info[2]
    year = str(date)[:4]
    day_of_year = date.timetuple().tm_yday - 1 #So that the year starts at day 0
    time_of_day = str(time)[:2] #get only the 00z, 06z etc. part
    
    days_offset = 2
    time_offset = dt.timedelta(days=days_offset)

            
    temp_files = sorted(glob('climatology_data/*t_near_surface.nc'))
    temp_data = xr.open_mfdataset(temp_files)
    daily_temp_data_mean = temp_data.groupby('time.dayofyear').mean(['time'])
        
    sphum_files = sorted(glob('climatology_data/*q_near_surface.nc'))
    sphum_data = xr.open_mfdataset(sphum_files)
    daily_sphum_data_mean = sphum_data.groupby('time.dayofyear').mean(['time'])
    
    all_times = ['00', '06', '12', '18']
            
    
    ds_before_files = []
    ds_after_files = []
    
    for t_o_d in all_times: #Loop through each time of day
        temp_file_str = '/home/disk/eos9/ERA5/hourly_pl/' + t_o_d + '/' + year + '.t.nc'
        sphum_file_str = '/home/disk/eos9/ERA5/hourly_pl/' + t_o_d + '/' + year + '.q.nc'

        ds_temp = xr.open_dataset(temp_file_str)
        ds_sphum = xr.open_dataset(sphum_file_str)

        full_ds = xr.merge([ds_temp, ds_sphum])
        
        new_datetime = dt.datetime.combine(date, dt.time(int(t_o_d), 0))

        try:
            ds_before_files.append(full_ds.sel(time=slice(new_datetime - time_offset, new_datetime)))

        except:
            pass

        try:
            ds_after_files.append(full_ds.sel(time=slice(new_datetime, new_datetime + time_offset)))

        except:
            pass
                
    ds_before = xr.concat(ds_before_files, dim='time')
    ds_after = xr.concat(ds_after_files, dim='time')
        
    ds_before_level = get_surface_level(ds_before)
    ds_after_level = get_surface_level(ds_after)
    
    if remove_season == True:
        day_before = day_of_year - days_offset
        day_after = day_of_year + days_offset
        
        if day_after > 365:
            day_after = 365
        else:
            pass
        
        if day_before < 0:
            day_before = 0
        else:
            pass

        final_temp_before = ds_before_level.t.mean(['time']) - daily_temp_data_mean.t[day_before,:,:]
        final_sphum_before = ds_before_level.q.mean(['time']) - daily_sphum_data_mean.q[day_before,:,:]
        final_ds_before = xr.merge([final_temp_before, final_sphum_before])
        
        final_temp_after = ds_after_level.t.mean(['time']) - daily_temp_data_mean.t[day_after,:,:]
        final_sphum_after = ds_after_level.q.mean(['time']) - daily_sphum_data_mean.q[day_after,:,:]
        final_ds_after = xr.merge([final_temp_after, final_sphum_after])
        
    else:
        final_ds_before = ds_before_level.mean(['time'])
        final_ds_after = ds_after_level.mean(['time'])
    
    return final_ds_before, final_ds_after


def grab_aht_data(datetime_info):
    '''Opens the AHT data corresponding to a given datetime
    Args:
        -datetime_info(list): List of datetime info from get_times_of_idx function
        
    Returns:
        -ds_time_sel(dataset): Xarray dataset of the AHT for that time
    
    '''
    
    date = datetime_info[0]
    time = datetime_info[1]
    date_time = datetime_info[2]
    year = str(date)[:4]
    day_of_year = date.timetuple().tm_yday - 1 #So that the year starts at day 0
    time_of_day = str(time)[:2] #get only the 00z, 06z etc. part
    
    rounded_doy = day_of_year - (day_of_year % 10)

    
    if rounded_doy == 360:  #End doy here is not plus 9 as we run out of days
        if int(year) in range(1980, 2030, 4): #Leap years are different
            end_doy = 365
        else:  #Not a leap year, but the end of the year
            end_doy = 364
    
    else:
        end_doy = rounded_doy + 9
    
    file_str = 'aht_calcs/' + str(year) + '/' + str(year) + '_' + str(time_of_day) +'z_' + str(rounded_doy) + '_' + str(end_doy)
        
    ds = xr.open_dataset(file_str)

    ds_time_sel = ds.sel(time=date_time)
    
    return ds_time_sel

    
    
def grab_temp_sphum_data(datetime_info, hour_offset):
    '''Creates a time-series of zonal-mean temp/sphum before and after a datetime
    Args:
        -datetime_info(list): List of datetime info from get_times_of_idx function
        
    Returns:
        -ds_before_level(dataset): Xarray dataset of the near-surface MSE for the two days before
        -ds_after_level(dataset): Xarray dataset of the near-surface MSE for the two days after
    
    '''
    
    date = datetime_info[0]
    time = datetime_info[1]
    date_time = datetime_info[2]
    year = str(date)[:4]
    day_of_year = date.timetuple().tm_yday - 1 #So that the year starts at day 0
    time_of_day = str(time)[:2] #get only the 00z, 06z etc. part
    
    time_offset = dt.timedelta(hours=hour_offset)
    
    datas = load_era5(int(year))
    
    new_datas = datas.chunk('auto')
        
    time_slice = (date_time + time_offset).to_pydatetime()

    try:
        datas_time = new_datas.sel(time=(date_time + time_offset))
    except:
        print('Time outside of bounds')
        
    return datas_time

def time_selector(vcomp, temp, sphum, geo_pot, time_info, orig_time):
    '''Accepts ERA5 data and some info on the time of interest and returns numpy arrays
    of data for that time.
    
    Args:
        -vcomp(Xarray datarray):Year of ERA5 vcomp data
        -temp(Xarray datarray):Year of ERA5 temp data
        -sphum(Xarray datarray):Year of ERA5 sphum data
        -geo_pot(Xarray datarray):Year of ERA5 geo_pot data
        -time_idx(datetime): Datetime of time series you want to grab
        -orig_time(datetime): Initial time that you are interested in
        
    Outputs:
        -vcomp_np(numpy array): Numpy array of vcomp for the one time_idx
        -temp_np(numpy array): Numpy array of temp for the one time_idx
        -sphum_np(numpy array): Numpy array of sphum for the one time_idx
        -geo_pot_np(numpy array): Numpy array of geo_pot for the one time_idx
        -time_point(datetime): Datetime of time series you grabbed
        -orig_time(datetime): Initial time that you are interested in
        
    '''
    vcomp_np = (vcomp.v.sel(time=time_info)).values
    temp_np = (temp.t.sel(time=time_info)).values
    sphum_np = (sphum.q.sel(time=time_info)).values
    geo_pot_np = (geo_pot.z.sel(time=time_info)).values
    
    time_point = vcomp.time.sel(time=time_info)
    
    return vcomp_np, temp_np, sphum_np, geo_pot_np, time_point, orig_time

def convert_to_xarray(datas):
    '''Works with time_selector to create xarray datasets of the data for a particular time
    
    Agrs:
        -datas(array): Array output from time_selector
        
    Outputs:
        -final_ds(xarray dataset): Input array turned into an xarray dataset
    
    '''
    
    vcomp_np = datas[0]
    temp_np = datas[1]
    sphum_np = datas[2]
    geo_pot_np = datas[3]
    time_point = datas[4]
    orig_time = datas[5]

    time_del = (pd.to_datetime(time_point.values) - orig_time) / pd.Timedelta('1 hour')
    
    final_ds = xr.Dataset(
                    data_vars = dict(
                        vcomp=(['level', 'latitude', 'longitude'], vcomp_np),
                        temp=(['level', 'latitude', 'longitude'], temp_np),
                        sphum=([ 'level', 'latitude', 'longitude'], sphum_np),
                        geo_pot=(['level', 'latitude', 'longitude'], geo_pot_np),
                    ),
                    coords=dict(
                        time_delta=time_del,
                        time=pd.to_datetime(time_point.values),
                        latitude=lats,
                        longitude=lons,
                        level=levels)
    )
    
    
    return final_ds


def find_nearest(array, value):
    '''Given an array and a value, returns the element of the aray closest to that value and its index
    
    Args:
        -array(array): Array of values
        -value(float): Value of interest
        
    Outputs:
        -out_value(float): Element of array closest to the input value
        -idx(int): idx of the output element in the aray
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    out_value = array[idx]
    return out_value, idx


def decorr_length_scale(datas):
    '''Accepts AHT data at one latitude and returns info on zonal-decorrelation length scale.
    
    Args:
        -data(array): Array of values of AHT at one latitude
        
    Outputs:
        -decorrs(array): Decorrelation length scale at each longitude
        -sum_corr_coefs(array): Average correlation with longitude
    '''
    #Find the correlation coefficients for each longitude with each other longitude
    corr_coefs = np.empty((720, 720))
    for i in range(720):
        for j in range(720):
            corr_coefs[i,j] = np.corrcoef(datas[:,i], datas[:,j])[0,1]
            
    #Now the messy part
    #We're going to basically put them all onto the same grid, so the longitude of interest
    #is at the center of the array
    #Then we can average them together too if we want
    dummy_corr_coefs = np.zeros((720, 720))
    decorrs = np.zeros(720)
    sum_corr_coefs = np.zeros(720)
    
    for i in range(720):
        corr_coefs_point = corr_coefs[i,:]

        if i<360:
            dummy_corr_coefs[i,360-i:360] = corr_coefs_point[:i]
            dummy_corr_coefs[i,360:] = corr_coefs_point[i:360+i]
            dummy_corr_coefs[i,:360-i] = corr_coefs_point[i-360:]

        elif i>360:
            dummy_corr_coefs[i,:360] = corr_coefs_point[i-360:i]
            dummy_corr_coefs[i,360:360-i] = corr_coefs_point[i:]
            dummy_corr_coefs[i,360-i:] = corr_coefs_point[:i-360]


        elif i==360:
            dummy_corr_coefs[i,:] = corr_coefs_point
        else:
            print('Problem!!!')

        #Add them together to create an average profile of decorrelation
        sum_corr_coefs += dummy_corr_coefs[i,:]
        
        #Find when the value drops below 1/e
        decorrs_end1 = find_nearest(dummy_corr_coefs[i,360:], 1/math.e)[1] + 360
        decorrs_end2 = find_nearest(dummy_corr_coefs[i,:360], 1/math.e)[1]

        #Now convert that to #grid points and average both sides
        idx_diff = (abs(decorrs_end1 - 360) + abs(decorrs_end2 - 360)) / 2

        decorrs[i] = idx_diff
        
    ave_corr_coefs = sum_corr_coefs / 720
        
    #Note, this returns decorrs in terms of grid points, not degrees
    #Divide by 2 to get degrees
    
    return decorrs, ave_corr_coefs


def plot_hist_and_gauss(axs, data, which_color, which_bins=60, scale_up=1, label='', plot_gaus=True):
    '''Takes data and an axis and plots a histogram and Gaussian fit of the data
    
    Args:
        -axs(matplotlib axis): Axis on which to plot things
        -data(array-like): Array of data to plot
        -which_color(str): Color to plot the data in
        -which_bins(int or array, default=60): Number of bins or the bins for the histograms
        -scale_up(int or float, default=1): Increase magnitude of data plotted
        -label(str): Label for the data
    
    Outputs:
        -Plots things
    '''
    
    # Bin it
    #Which bins can either be an int (# of bins) or the bins themselves
    n_bins, bin_edges = np.histogram(data, which_bins) 
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n_bins/float(n_bins.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    axs.bar(bin_middles, scale_up * bin_probability, width=bin_width,
            color=which_color, alpha=0.3, label=label)

    # Fit to normal distribution
    (mu, sigma) = sc.norm.fit(data)
    # The pdf should not normed anymore but scaled the same way as the data
    gaus = sc.norm.pdf(bin_middles, mu, sigma) * bin_width
    
    if plot_gaus == True:
        axs.plot(bin_middles, gaus * scale_up, color=which_color, linewidth=2)
    else:
        pass
    
    
    
def calc_strm_funct(datas):
    '''Calculates the meridional overturning streamfunction from monthly data in pressure coordinates by month
    
    Args:
        datas(Xarray dataset)- An Xarray dataset from ERA5 output
        
    Output:
        strm_fnct_data(Xarray DataArray) - Meridional streamfunction
    '''
    
    time=datas.time
    lats=datas.latitude
    lons=datas.longitude
    levels=datas.level
    
    zon_norms = np.load('../Calculate_AHT/zonal_norms.npy') #Dims (level, lat, lon)
    
    #Divide weights by g to get the units right
    weights = np.load('../Calculate_AHT/aht_weights.npy')/ g #Dims (level, lat, lon)
    weights[np.isnan(weights)] = 0
    weights_zon_mean = np.nanmean(weights, axis=2) #Dims (level, lat)
    
    geom_multiplier = 2 * np.pi * a * np.cos(lats.values*np.pi/180) #Dims (lat)
    
    vcomp = datas.v #Dims (time, level, lat, lon)
    vcomp_zon_mean = np.nansum(vcomp * zon_norms[None,:,:,:], axis=3) #Dims (time, level, lat)

    mass_flux = vcomp_zon_mean * weights_zon_mean[None,:,:] * geom_multiplier[None,None,:] #Dims (time, level, lat)

    vcomp_baro = np.nansum(mass_flux, axis=1) / ((geom_multiplier)*np.nansum(weights_zon_mean, axis=0))[None,:] #Dims (time, lat)

    vcomp_corrected = vcomp_zon_mean - vcomp_baro[:, None,:] #Dims (time, level, lat)

    mass_flux_corrected = vcomp_corrected * weights_zon_mean[None,:,:] * geom_multiplier[None,None,:] #Dims (time, level, lat)
    mass_flux_corrected_reverse = mass_flux_corrected[:,::-1,:]
    
    strm_fnct = np.nancumsum(mass_flux_corrected, axis=1)

    strm_fnct_da = xr.DataArray(data=strm_fnct,
                                dims=['time', 'level', 'latitude'],
                            coords=dict(
                                time=time,
                                latitude=lats,
                                level=levels)
        )
        
    return(strm_fnct_da)