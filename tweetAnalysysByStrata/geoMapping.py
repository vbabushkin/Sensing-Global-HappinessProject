__author__ = 'vahan'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import scipy.stats as stats
import math
RESOLUTION=512j
BANDWIDTH=0.0015 # use 0.05 for ordinary heatmap
PERCENTILE=80
BASE=2

#England
# MINLAT=50#-90 #lowest left corner latitude
# MAXLAT=60 #90 #lowest right corner latitude
# MINLON=-11#-180 #lowest left corner longitude
# MAXLON=2  #180 #lowest right corner longitude

#Wholeworld
MINLAT=-90 #lowest left corner latitude
MAXLAT=90 #lowest right corner latitude
MINLON=-180 #lowest left corner longitude
MAXLON=180 #lowest right corner longitude

fig = plt.figure()

m = Basemap(projection='cyl',llcrnrlat=MINLAT,urcrnrlat=MAXLAT,llcrnrlon=MINLON,urcrnrlon=MAXLON,resolution='c')
m.shadedrelief()

#m.drawcoastlines()
#labels = [left,right,top,bottom]
m.drawparallels(np.arange(-90.,90.,20.),labels=[True,False,True,False],linewidth=0.0)
m.drawmeridians(np.arange(0.,360.,20.),labels=[True,False,False,True],linewidth=0.0)
m.drawmapboundary(fill_color='white')
plt.title('Geolocated Tweets Heatmap')

#ADD HEATMAP TO THE PLOT


with open('collectedTweetsCoordinates.txt') as f:
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
kde.set_bandwidth(BANDWIDTH)
#kde.set_bandwidth(bw_method=kde.factor / 3.)


#for ordinary plot
#z = kde(grid_coords.T)


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


img=ax.imshow(z,origin='lower',extent=(-180,180,-90,90),alpha=0.5)




# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.2)


cb = fig.colorbar(img, cax=cax)

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

cb.set_label('number of tweets')





#

#comment to suspend output of scatter markers
ax.scatter(rvs[:,0],rvs[:,1],s=0.9,marker='o',alpha=0.7,color='white')



plt.show()






