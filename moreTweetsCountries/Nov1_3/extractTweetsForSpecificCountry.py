__author__ = 'vahan'


#http://nbviewer.ipython.org/github/davidrpugh/cookbook-code/blob/master/notebooks/chapter14_graphgeo/06_gis.ipynb
#shape file is taken from here: file:///home/vahan/PycharmProjects/tweetAnalysis/borders/ne_10m_admin_0_countries.README.html
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as col
import json
import time
import ast
from mpl_toolkits.basemap import Basemap
import fiona
import shapely.geometry as geom
from shapely.geometry import Point, shape
from descartes import PolygonPatch


# natural earth data
countries = fiona.open("/home/vahan/PycharmProjects/tweetAnalysis_v1/borders/ne_10m_admin_0_countries.shp")
#print countries.schema


countryName = ["India", "Nigeria", "Pakistan", "Kenya", "Ghana", "Cameroon", "Nepal", "Zimbabwe", "Sudan", "Uganda", "Senegal", "Yemen", "Malawi", "Haiti", "Bangladesh", "Mozambique", "Zambia", "Angola", "Indonesia", "Philippines", "Thailand", "Colombia", "South Africa", "Egypt", "Paraguay", "Ecuador", "Guatemala", "Iraq", "El Salvador", "Nicaragua","China", "Honduras", "Jamaica", "Morocco", "Algeria", "Mongolia", "Botswana", "Vietnam", "Tunisia", "Bolivia", "Namibia", "Maldives", "Uzbekistan"]


REGION_TO_EXTRACT='World'

# REGION_TO_EXTRACT='Europe'
# COUNTRY_TO_EXTRACT='England'

#Create a basemap map showing the desired region.
#http://isithackday.com/geoplanet-explorer/index.php?woeid=24865675
if (REGION_TO_EXTRACT=='Africa'):
    m = Basemap(llcrnrlon=-23.03,
                llcrnrlat=-37.72,
                urcrnrlon=55.20,
                urcrnrlat=40.58)
    #We select the countries.
    region = [c for c in countries if c['properties']['CONTINENT'] == REGION_TO_EXTRACT]

if (REGION_TO_EXTRACT=='Europe'):
    m = Basemap(llcrnrlon=-31.266001,
            llcrnrlat=27.636311,
            urcrnrlon=39.869301,
            urcrnrlat=81.008797)
    #We select the countries.
    region = [c for c in countries if c['properties']['CONTINENT'] == REGION_TO_EXTRACT]

if (REGION_TO_EXTRACT=='World'):
    #for the whole world
    region=countries
    m = Basemap(llcrnrlon=-180,
            llcrnrlat=-90,
            urcrnrlon=180,
            urcrnrlat=90)



#We need to convert the geographical coordinates of the countries' borders in map coordinates, so that we can display then in basemap.
def _convert(poly, m):
    if isinstance(poly, list):
        return [_convert(_, m) for _ in poly]
    elif isinstance(poly, tuple):
        return m(*poly)

for _ in region:
    _['geometry']['coordinates'] = _convert(
        _['geometry']['coordinates'], m)



t0 = time.time()



for line in open('filteredGeolocatedTweets_Nov_1_3', 'r'):
        tweet_json = ast.literal_eval(line)
        tweet_coord= tweet_json['coordinates']['coordinates']
        #coordinates of our point to locate
        point = Point(tweet_coord[0],tweet_coord[1])
        for feature in region:
            if feature['properties']['NAME']in countryName:
                if shape(feature['geometry']).contains(point):
                    filename1="tweetsByCountriesCoordinatesNov1_3/extractedTweetsFrom"+feature['properties']['NAME']+".txt"
                    filename2="tweetsByCountriesFullNov1_3/extractedTweetsFrom"+feature['properties']['NAME']+".txt"
                    #print filename1
                    coordFile = open(filename1, 'a')
                    coordFile.write(str(tweet_coord[0])+"\t"+str(tweet_coord[1])+"\n")
                    coordFile.close
                    coordFile = open(filename2, 'a')
                    coordFile.write(line)
                    coordFile.close

t1 = time.time()

total = t1-t0

print "TOTAL TIME REQUIRED: "
print total
#


#THIS CODE IS FOR EXTRACTING TWEETS IF THE NAME OF COUNTRY IS SPECIFIED
#
#
# #coordinates of our point to locate
# point = Point(3.65,51.3)
#
# for feature in region:
#     if shape(feature['geometry']).contains(point):
#           print feature['properties']['NAME']
#
#
# t1 = time.time()
# total = t1-t0
# print "total time to find the point:"
# print total




#SOME TEST CODE
#
# #which features we have
# countries[0]['properties'].keys()
#
#
# name=u'Netherlands'
#
#
# #find netherlands
# for country in europe:
#     if country['properties']['NAME']==name:
#         desiredCountry=country
#
# coords=desiredCountry['geometry']['coordinates']
#
#
#
# #Now, we create a basemap map showing the African continent.
# m = Basemap(llcrnrlon=3,
#             llcrnrlat=51,
#             urcrnrlon=4,
#             urcrnrlat=51)
#
#
#
# polygon=desiredCountry['geometry']
# print polygon.bounds
#
# # #test the function
# # poly1=coords[3][0]
# #
# # x0=3.65
# #
# # y0=51.3
# #
# # point_in_poly(x0,y0,poly1)
#
#
# # for coordslistGroup in coords:
# #     for coordsList in coordslistGroup:
# #
# #
# #
# #
# #
# #
# #Now, we create a basemap map showing the Europe
# m = Basemap(llcrnrlon=12.427394924000097,#-23.03,
#             llcrnrlat=-69.87682044199994,#-37.72,
#             urcrnrlon=12.546820380000057,#55.20,
#             urcrnrlat=-70.06240800699987)#40.58)
