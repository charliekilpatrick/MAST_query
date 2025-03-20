import numpy as np
import os
import glob
import copy
import pandas
import sys

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.table import Column
from astropy.table import vstack

# For transforming sky ellipses into polygons on sky
from regions import EllipseSkyRegion

# Checking mast and downloading SkyView images
import astroquery
from astroquery.mast import Observations
from astroquery.skyview import SkyView

# Shapely stuff for quick crossmatching in 2D
import shapely
from shapely.geometry import Polygon 
from shapely.strtree import STRtree

# Plotting stuff
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

def import_gwgc(trim=True):
    t = ascii.read('data/GWGC.csv')

    # Trim rows with missing key data in table
    if trim:
        mask = t['absbmag']!='NULL'
        t = t[mask]

        mask = t['tt']!='NULL'
        t = t[mask]

        mask = t['a']!='NULL'
        t = t[mask]

        mask = t['b']!='NULL'
        t = t[mask]

        mask = t['pa']!='NULL'
        t = t[mask]

        t['absbmag'] = t['absbmag'].astype(float)
        t['tt'] = t['tt'].astype(float)
        t['a'] = t['a'].astype(float)
        t['b'] = t['b'].astype(float)
        t['pa'] = t['pa'].astype(float)

        # Adding inclination row using method described in Ho+2011:
        # https://arxiv.org/pdf/1111.4605, eq. 26 with q0=0.2
        ellipticity = np.sqrt(1.0 - t['b']**2 / t['a']**2)
        q0 = 0.2
        cos2i = ((1-ellipticity)**2 - q0**2)/(1-q0**2)
        incl = np.sqrt(np.arccos(cos2i)) * 180.0/np.pi
        t['incl'] = incl

    return(t)

def trim_galaxy_catalog(catalog, cuts):
    
    for key in cuts.keys():
        mask = catalog[key] > cuts[key][0]
        mask = mask & (catalog[key] < cuts[key][1])

        catalog = catalog[mask]

    return(catalog)

def make_galaxy_polygons(catalog):
    galaxies = []
    for row in catalog:

        width = row['a']/60.0
        height = row['b']/60.0
        angle = row['pa']

        # Start with defining sky region with EllipseSkyRegion
        coord = SkyCoord(row['ra0'], row['dec0'], unit='deg')
        reg = EllipseSkyRegion(coord, row['a']*u.arcmin,
            row['b']*u.arcmin, (90.0+row['pa'])*u.deg)

        # Create dummy WCS object so we can convert into a pixel region
        wcs = WCS({'NAXIS': 2, 'CRPIX1':50.0, 'CRPIX2':50.0, 
            'CDELT1':0.0001, 'CDELT2':0.0001, 'CRVAL1':row['ra0'], 
            'CRVAL2':row['dec0'], 'CTYPE1':'RA---TAN', 'CTYPE2':'DEC--TAN'})
        pixreg = reg.to_pixel(wcs)

        # Now create an ellipse with region.to_artist() and get vertices in pix
        ellipse = pixreg.as_artist()
        vertices = ellipse.get_verts()

        # Finally, convert back to sky coordinates and polygon in those coords
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        coords = pixel_to_skycoord(xs, ys, wcs)
        vertices = [[c.ra.degree, c.dec.degree] for c in coords]

        ellipse = Polygon(vertices)

        galaxies.append(ellipse)

    return(galaxies)

def get_MAST_polygons(df):

    polys=[]
    indices = []
    for idx, row in df.iterrows():
        polystr = row['s_region']
        thispoly = polystr.split('POLYGON')[1:]
        for polystr in thispoly:
            # Need a try/except as some of the MAST polygons are poorly formatted
            try:
                polys.append(shapely.Polygon(np.array(polystr.split()).reshape((-1,2)).astype(float)))
                indices.append(idx)
            except ValueError:
                continue

    return(polys, indices)

def trim_hst_table(hst):

    # Ignore "detection" rows and nan filter rows
    hst = hst.loc[lambda df: df['filters']!='detection']
    hst = hst.loc[lambda df: df['filters']!='nan']

    # Ignore grating observations - spectroscopic
    hst = hst.loc[~hst['filters'].str.startswith('G', na=True)]

    # Ignore "FR" filters
    hst = hst.loc[~hst['filters'].str.startswith('FR', na=True)]

    # Ignore narrow filters
    hst = hst.loc[~hst['filters'].str.endswith('N', na=True)]

    # Ignore medium filters
    hst = hst.loc[~hst['filters'].str.endswith('M', na=True)]

    # Ignore all other filters that do not start with "F"
    hst = hst.loc[hst['filters'].str.startswith('F', na=False)]

    return(hst)

def get_all_galaxy_filts(galaxy_tree, gwgc, hst, polys, indices):

    all_data = []
    for i in np.arange(len(polys)):
        result = galaxy_tree.query(polys[i])
        objnames = []
        if len(result)>0:
            for jj in result:
                objnames.append(gwgc[jj]['name'])

        # Can ignore observations that dont cover anything
        if len(objnames)==0: continue

        filt = hst.loc[indices[i]]['filters']

        all_data.append((filt, objnames))

    # Now we need to invert into a list of objects with filters
    all_names = {}

    for row in all_data:
        filt = row[0]
        names = row[1]

        for n in names:
            if n.strip():
                if n not in all_names.keys():
                    all_names[n]=[filt]
                elif filt not in all_names[n]:
                    all_names[n].append(filt)

    return(all_names)

def do_test_ngc(gwgc, name='NGC3631'):
    gwgc = gwgc[gwgc['name']==name]

    polys = make_galaxy_polygons(gwgc)
    coord = SkyCoord(gwgc[0]['ra0'], gwgc[0]['dec0'], unit='deg')

    reg = EllipseSkyRegion(coord, gwgc[0]['a']*u.arcmin,
        gwgc[0]['b']*u.arcmin, (90.0+gwgc[0]['pa'])*u.deg)

    paths = SkyView.get_images(position=SkyCoord(gwgc[0]['ra0'],
        gwgc[0]['dec0'], unit='deg'),
        survey=['DSS'])

    wcs = WCS(paths[0][0].header)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)

    ax.imshow(paths[0][0].data, cmap='gray')
    ax.set_xlabel(r'RA')
    ax.set_ylabel(r'Dec')

    pixreg = reg.to_pixel(wcs)
    pixreg.plot(ax=ax, color='red', lw=2.0)

    if not os.path.exists('output'):
        os.makedirs('output')

    plt.savefig(f'output/{name}.png')

if __name__=="__main__":

    gwgc = import_gwgc(trim=True)

    cuts={'dist': [0.0, 20.0], # <20 Mpc
          'absbmag': [-np.inf, -18.0], # M_B < -18.0 mag
          'tt': [-3.0, np.inf], # Galaxy Type > 0
          'Ebv': [0.0, 1.0/3.1], # A_V < 1.0 mag
          'incl': [0.0, 72.66], # No galaxies closer than 73 deg to edge on
          'a': [0.0, 4.0], # Semi-major axis is <4.0 arcmin
          }

    # Check for these filters.  Filters with / are suitable if only 1 exists
    check_filters = ['F275W/F336W','F555W/F606W','F814W']

    gwgc = trim_galaxy_catalog(gwgc, cuts)

    if True:
        do_test_ngc(gwgc)

    # Create a Sort-Tile-Recursive tree for crossmatching galaxy polygons
    galaxy_polygons = make_galaxy_polygons(gwgc)
    galaxy_tree = STRtree(galaxy_polygons)

    # For some reason this table can only be read in with pandas
    hst = pandas.read_csv('data/MAST_HST_20250319.csv')

    # Trim HST table to ignore filters that are not of interest
    hst = trim_hst_table(hst)

    polys, indices = get_MAST_polygons(hst)

    galaxy_obs_dict = get_all_galaxy_filts(galaxy_tree, gwgc, hst, polys, indices)

    print('Galaxy observations still needed:')
    total_orbits = 0
    total_objects = 0
    total_any_obs = 0
    obs_list = []
    for row in gwgc:
        need_filts=[]

        for filt in check_filters:
            filts = filt.split('/')
            skip = False
            if row['name'] in galaxy_obs_dict.keys():
                for f in filts:
                    if f in galaxy_obs_dict[row['name']]: skip=True

            if not skip: need_filts.append(filts[0])

        if len(need_filts)>0:
            print(row['name'],' '.join(need_filts))
            total_objects += 1

        if len(need_filts)<len(check_filters):
            total_any_obs += 1

        total_orbits += len(need_filts)
        obs_list.extend(need_filts)

    print('\n\nSummary:')
    print(f'Total number of galaxies checked: {len(gwgc)}')
    print(f'Total number of galaxies with any observation in {check_filters}: {total_any_obs}')
    print(f'Total number of galaxies to observe: {total_objects}')
    print(f'Total number of orbits to observe: {total_orbits}')
    print(f'Total number of orbits needed for each filter:')
    for filt in check_filters:
        f = filt.split('/')[0]
        print(f'{f} count: {obs_list.count(f)}')





