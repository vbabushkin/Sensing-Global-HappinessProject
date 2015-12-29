__author__ = 'vahan'
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import scipy.stats as stats
import math
import time
import ast
from mpl_toolkits.basemap import Basemap
import fiona
import shapely.geometry as geom
from shapely.geometry import Point, shape


RESOLUTION=256j
BANDWIDTH=0.015 # use 0.05 for ordinary heatmap
PERCENTILE=30
BASE=2


#load countries shapefile with borders
# natural earth data
countries = fiona.open("borders/ne_10m_admin_0_countries.shp")

#print countries.schema

#specify country name, 'World' for the whole world
#name=u'United Arab Emirates'
#name='World'


name="Germany"
#specify kde to be used 'ordnary' or 'logarithmic'use 'Ord' and 'Log'
kdeUsed='Ord'



if name!='World':
    #file containing the coordinates for heatmap
    filename1="oct1516/tweetsByCountriesCoordinates/extractedTweetsFrom"+name+".txt"

    #find netherlands
    for country in countries:
        if country['properties']['NAME']==name:
            desiredCountry=country


    #country = countries.next()
    print "country name :",desiredCountry['properties']['NAME']

    bounds=shape(desiredCountry['geometry']).bounds
    print "bounds:",bounds

    #Selected Country
    MINLAT=bounds[1]#-90 #lower left corner latitude
    MAXLAT=bounds[3] #90 #lower right corner latitude
    MINLON=bounds[0]#-180 #upper left corner longitude
    MAXLON=bounds[2] #180 #upper right corner longitude
else:
    #Wholeworld
    MINLAT=-90 #lowest left corner latitude
    MAXLAT=90 #lowest right corner latitude
    MINLON=-180 #lowest left corner longitude
    MAXLON=180 #lowest right corner longitude
    filename1="collectedTweetsCoordinates.txt"



fig = plt.figure()

m = Basemap(projection='cyl',llcrnrlat=MINLAT,urcrnrlat=MAXLAT,llcrnrlon=MINLON,urcrnrlon=MAXLON,resolution='c')
m.shadedrelief()

#m.drawcoastlines()
#labels = [left,right,top,bottom]
m.drawparallels(np.arange(round(MINLAT,2),round(MAXLAT,2),abs(round((MAXLAT-MINLAT)/10,2))),labels=[True,False,True,False],linewidth=0.1,fontsize=12)
m.drawmeridians(np.arange(round(MINLON,2),round(MAXLON,2),abs(round((MAXLON-MINLON)/5,2))),labels=[True,False,False,True],linewidth=0.1,fontsize=12)
m.drawmapboundary(fill_color='white')
m.drawcountries(linewidth=1)
plt.title('Tweets Ordinary Heatmap for '+name, fontsize=30)
#set for the remaining text
plt.rcParams.update({'font.size': 15})
#ADD HEATMAP TO THE PLOT




with open(filename1) as f:
    w, h = [float(x) for x in f.readline().split()] # read first line
    pts = []
    for line in f: # read rest of lines
        pts.append([float(x) for x in line.split()])
    data=np.array(pts)



x=(data[:,0]>=MINLON)&(data[:,0]<=MAXLON)
data=data[x,:]

y=(data[:,1]>=MINLAT)&(data[:,1]<=MAXLAT)
data=data[y,:]



rvs=data[1:len(data)]
ax = fig.add_subplot(111)

x_flat = np.r_[MINLON:MAXLON:RESOLUTION]

y_flat = np.r_[MINLAT:MAXLAT:RESOLUTION]

x,y = np.meshgrid(x_flat,y_flat)

grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)




kde = stats.kde.gaussian_kde(rvs.T)
#kde.set_bandwidth(bw_method='silverman')

if kdeUsed=='Ord':
    #UNCOMMENT IF USING BANDWIDTH
    #kde.set_bandwidth(BANDWIDTH)

    #UNCOMMENT TO SET AUTOMATIC BANDWIDTH
    #kde.set_bandwidth(bw_method=kde.factor / 1.2)


    #for ordinary plot
    z = kde(grid_coords.T)

elif kdeUsed=='Log':
    # use logarithmic plot
    z=[]

    old_z = kde(grid_coords.T)

    for item in old_z:
        if(item<=0):
            item=item+1e-300
        z.append(math.log(item)/math.log(BASE))


z=np.array(z)

z = z.reshape(RESOLUTION.imag,RESOLUTION.imag)




#set percentile of kde values to prune
p=np.percentile(z, PERCENTILE)
z[z<p]=None


img=ax.imshow(z,origin='lower',extent=(MINLON,MAXLON,MINLAT,MAXLAT),alpha=0.5)

#

#comment to suspend output of scatter markers
ax.scatter(rvs[:,0],rvs[:,1],s=0.9,marker='o',alpha=0.7,color='white')


# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.2)


cb = fig.colorbar(img, cax=cax)
ax = cb.ax



yticks= cb.ax.get_yticks()
print yticks

diff= yticks[len(yticks)-1]-yticks[0]
print diff




mlow=int(len(data) *(1-diff) )                   # colorbar min value
mhigh=int(len(data) *diff )                       # colorbar max value
ticks=[mlow]

for i in range(1,len(yticks)-1):
    ticks.append(int((i*(mhigh-mlow)/len(yticks) + mlow)*diff))

ticks.append(mhigh)


# m1=int((1*(m7-m0)/8.0 + m0)*diff)              # colorbar mid value
# m2=int((2*(m7-m0)/8.0 + m0)*diff)              # colorbar mid value
# m3=int((3*(m7-m0)/8.0 + m0)*diff)              # colorbar mid value
# m4=int((4*(m7-m0)/8.0 + m0)*diff)              # colorbar mid value
# m5=int((5*(m7-m0)/8.0 + m0)*diff)              # colorbar mid value
# m6=int((6*(m7-m0)/8.0 + m0)*diff)              # colorbar mid value


#cb.ax.get_yaxis().set_ticks([])
#cb.ax.set_yticks([m0,math.log(m1),math.log(m2),math.log(m3),math.log(m4)])



# ticks=[m0,m1,m2,m3,m4, m5,m6, m7]

print ticks




cb.ax.set_yticklabels(ticks)

cb.set_label('number of tweets',fontsize=25)


mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.savefig('heatmap'+name+kdeUsed+'.png',transparent=True, bbox_inches='tight', pad_inches=0)

plt.show()
#
# ##########################################################################################################
# __author__ = 'Masdar'
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
#
#
# with open(filename1) as f:
#     w, h = [float(x) for x in f.readline().split()] # read first line
#     pts = []
#     for line in f: # read rest of lines
#         pts.append([float(x) for x in line.split()])
#     data=np.array(pts)
#     print data[1:10]
#
# rvs=data[1:len(data)]
# print len(data)
# kde = stats.kde.gaussian_kde(rvs.T)
#
# # Regular grid to evaluate kde upon
# x_flat = np.r_[rvs[:,0].min():rvs[:,0].max():128j]
# y_flat = np.r_[rvs[:,1].min():rvs[:,1].max():128j]
# x,y = np.meshgrid(x_flat,y_flat)
# grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
# print grid_coords
# z = kde(grid_coords.T)
# z = z.reshape(128,128)
# imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))
# fig = plt.figure()
# imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))
# ax = fig.add_subplot(111)
# ax.scatter(rvs[:,0],rvs[:,1],s=0.5,marker='o',alpha=0.15,color='white')
#
# plt.show()
#
#
#
#
#
