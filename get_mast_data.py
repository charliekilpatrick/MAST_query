import astroquery
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.io import ascii
from astropy.table import Column, vstack
import numpy as np
import progressbar
import os
import datetime

def get_MAST_imaging_data(telescope='hst', redo=False):

    telescope = telescope.lower()
    if not os.path.exists('data/mast'):
        os.makedirs('data/mast')
    if not os.path.exists(f'data/mast/{telescope}'):
        os.makedirs(f'data/mast/{telescope}')

    if telescope=='hst':
        obs_collection = ['HST']
        instrument_name=['ACS','ACS/HRC','WFC3/IR','WFC3/UVIS','WFPC2',
            'WFPC2/PC','WFPC2/WFC','ACS/WFC']

    incr=1.0
    iters = int(360./incr)
    for i in np.arange(iters):

        curr_ra = i*incr
        if os.path.exists(f'data/mast/{telescope}/{curr_ra}.csv') and not redo:
            continue

        print('starting query')
        obsTable = Observations.query_criteria(
            dataproduct_type=["image"],
            obs_collection=obs_collection,
            intentType=['science'],
            instrument_name=instrument_name,
            s_ra=[curr_ra,curr_ra+incr])

        outfile = f'data/mast/{telescope}/{curr_ra}.csv'
        obsTable.write(outfile, format='csv', overwrite=True)

        outstr = f'RA={curr_ra}->{curr_ra+1.0}: {len(obsTable)} {telescope} records'
        outstr += f' outfile={outfile}'
        print(outstr)

def combine_MAST_data(telescope='HST'):

    tel_upper = telescope.upper()
    datestr = datetime.datetime.utcnow().strftime('%Y%m%d')

    output = open(f'data/MAST_{tel_upper}_{datestr}.csv','w')

    telescope = telescope.lower()
    for i in np.arange(360):
        i=float(i)
        file = f'data/mast/{telescope}/{i}.csv'
        print(file)
        with open(file, 'r') as f:
            for j,line in enumerate(f):
                # Skip header unless it is for first file
                if j<1 and i>0: continue
                output.write(line)

    output.close()

    print('done with combination')


if __name__=="__main__":

    get_MAST_imaging_data(telescope='HST', redo=False)
    combine_MAST_data(telescope='HST')
