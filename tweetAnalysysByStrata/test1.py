"""
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.basemap import Basemap
    have_basemap = True
except ImportError:
    have_basemap = False


def plotmap():
    # create figure
    fig = plt.figure(figsize=(8,8))
    # set up orthographic map projection with
    # perspective of satellite looking down at 50N, 100W.
    # use low resolution coastlines.
    map = Basemap(projection='ortho',lat_0=50,lon_0=-100,resolution='l')
    # lat/lon coordinates of five cities.
    lats=[40.02,32.73,38.55,48.25,17.29]
    lons=[-105.16,-117.16,-77.00,-114.21,-88.10]
    cities=['Boulder, CO','San Diego, CA',
            'Washington, DC','Whitefish, MT','Belize City, Belize']
    # compute the native map projection coordinates for cities.
    xc,yc = map(lons,lats)
    # make up some data on a regular lat/lon grid.
    nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
    lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
    lons = (delta*np.indices((nlats,nlons))[1,:,:])
    wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
    mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
    # compute native map projection coordinates of lat/lon grid.
    # (convert lons and lats to degrees first)
    x, y = map(lons*180./np.pi, lats*180./np.pi)
    # draw map boundary
    map.drawmapboundary(color="0.9")
    # draw graticule (latitude and longitude grid lines)
    map.drawmeridians(np.arange(0,360,30),color="0.9")
    map.drawparallels(np.arange(-90,90,30),color="0.9")
    # plot filled circles at the locations of the cities.
    map.plot(xc,yc,'wo')
    # plot the names of five cities.
    for name,xpt,ypt in zip(cities,xc,yc):
        plt.text(xpt+100000,ypt+100000,name,fontsize=9,color='w')
        # contour data over the map.
    cs = map.contour(x,y,wave+mean,15,linewidths=1.5)
    # draw blue marble image in background.
    # (downsample the image by 50% for speed)
    map.bluemarble(scale=0.5)

def plotempty():
    # create figure
    fig = plt.figure(figsize=(8,8))
    fig.text(0.5, 0.5, "Sorry, could not import Basemap",
        horizontalalignment='center')

if have_basemap:
    plotmap()
else:
    plotempty()
plt.show()

"""
#make a scatter plot with varying color and size arguments
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

# load a numpy record array from yahoo csv data with fields date,
# open, close, volume, adj_close from the mpl-data/example directory.
# The record array stores python datetime.date as an object array in
# the date column
datafile = cbook.get_sample_data('goog.npy')
r = np.load(datafile).view(np.recarray)
r = r[-250:] # get the most recent 250 trading days

delta1 = np.diff(r.adj_close)/r.adj_close[:-1]

# size in points ^2
volume = (15*r.volume[:-2]/r.volume[0])**2
close = 0.003*r.close[:-2]/0.003*r.open[:-2]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.75)

#ticks = arange(-0.06, 0.061, 0.02)
#xticks(ticks)
#yticks(ticks)


ax.set_xlabel(r'$\Delta_i$', fontsize=20)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=20)
ax.set_title('Volume and percent change')
ax.grid(True)

plt.show()

"""

import numpy as np



with open('collectedTweetsCoordinates.txt') as f:
    w, h = [float(x) for x in f.readline().split()] # read first line
    pts = []
    for line in f: # read rest of lines
        pts.append([float(x) for x in line.split()])
    #print pts[1:10]
    #print pts[0:10][0]
    #print pts[0:10][1]
    data=np.array(pts)
    #print data[1:10]


""""
    data=np.transpose(pts[1:100])
    print data
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:32:51 2011

@author: endolith@gmail.com

"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import math

"""
# Create some dummy data
rvs = np.append(stats.norm.rvs(loc=2,scale=1,size=(200,1)),
    stats.norm.rvs(loc=1,scale=3,size=(200,1)),
    axis=1)
"""
rvs=data[1:100000]

kde = stats.kde.gaussian_kde(rvs.T)

# Regular grid to evaluate kde upon
# x_flat = np.r_[rvs[:,0].min():rvs[:,0].max():128j]
#
# y_flat = np.r_[rvs[:,1].min():rvs[:,1].max():128j]
#
# x,y = np.meshgrid(x_flat,y_flat)



x_flat = np.r_[-180:180:128j]

y_flat = np.r_[-90:90:128j]

x,y = np.meshgrid(x_flat,y_flat)

grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)


# #use logarithmic plot
# z=[]

# old_z = kde(grid_coords.T)

# for item in old_z:
#     z.append(math.log(item))
#
# z=np.array(z)


z = kde(grid_coords.T)
z = z.reshape(128,128)
#imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))

imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(-180,180,-90,90))
x_ticks=np.arange(-180,180,20)
y_ticks=np.arange(-90,90,20)
plt.xticks(x_ticks,fontsize=9)
plt.yticks(y_ticks,fontsize=9)

fig = plt.figure()
# imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))
imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(-180,180,-90,90))
ax = fig.add_subplot(111)
ax.scatter(rvs[:,0],rvs[:,1],s=0.5,marker='o',alpha=0.15,color='white')

plt.show()