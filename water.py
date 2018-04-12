#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-12 09:12
#

"""

""" 
import json, fiona
# 第一步，读取标注的矢量文件，存储为earthengine的FeatureCollection类型
shp_file = '/mnt/win/water_paper/training_data/TM/feature_shp/water-L5134036.shp'

vector = fiona.open(shp_file, 'r')
shp_proj = vector.crs.values()[0]
geojson_list = []
for feature in vector:
    # create a shapely geometry
    # this is done for the convenience for the .bounds property only
    # feature['geoemtry'] is in Json format
    geojson = feature['geometry']
    geojson_list.append(geojson)


import ee
ee.Initialize()
features = []
for i in range(len(geojson_list)):
    type = geojson_list[i]['type']
    if type == 'Polygon':
        # Earth Engine's geometry constructors build geodesic geometries by default. To make a planar geometry, constructors have a geodesic parameter that can be set to false
        # 创建geometry时，默认是地理坐标；如果不是地理坐标，则geodesic参数设置为False
        record = ee.Feature(ee.Geometry.Polygon(geojson_list[i]['coordinates'], proj=shp_proj,geodesic=False)).set('class','water')
    elif type == 'MultiPolygon':
        # continue
        record = ee.Feature(ee.Geometry.MultiPolygon(geojson_list[i]['coordinates'], proj=shp_proj,geodesic=False),{'class':'water'})
    elif type == 'Point':
        record = ee.Feature(ee.Geometry.Point(geojson_list[i]['coordinates'], proj=shp_proj,geodesic=False), {'class':'water'})
    features.append(record)

# create a FeatureCollection from the list and print it.
FCfromList = ee.FeatureCollection(features)

# 第二步，读取标注的矢量文件对应的遥感影像,
# method 1 通过dataid进行过滤,但可能因为数据版本原因找不到对应影像， 如 dataid= "LT51340362005235BJC00"，gee中LT51340362005235BJC01
dataid='LT51340362005235BJC01'
imagecol = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA').filterMetadata('LANDSAT_SCENE_ID', 'equals', dataid)
image = ee.Image(imagecol.first())

# method 2 通过标注矢量文件范围，和时间
roi = ee.Geometry.MultiPolygon(geojson_list[0]['coordinates'], proj=shp_proj,geodesic=False)
image = ee.Image(ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA').filterBounds(roi).filterDate('2005-08-01','2005-08-30').sort('CLOUD_COVER').first())
# other metadata properties: 'WRS_PATH': 134,  'WRS_ROW': 36,

# 第三步，计算各种water index
# example: evi = image.expression('2.5*((NIR - RED)/(NIR + 6*RED-7.5*BLUE+1))',{'NIR':image.select('B5'), 'RED':image.select('B4'), 'BLUE':image.select('B2')})
# for landsat45: band1:blue, band2:green, band3:red, band4:nir, band5:swir5, band7:swir7
# NDWI = (band4 - band5) / (band4 + band5)
# MNDWI = (band2 - band5) / (band2 + band5)
# EWI = (band2 - band4 - band5) / (band2 + band4 + band5)
# NEW = (band1 - band7) / (band1 + band7)
# NDWI_B = (band1 - band4) / (band1 + band4)
# AWElnsh = 4 * (band2 - band5) - (0.25 * band4 + 2.75 * band7)

NDWI = image.expression('(NIR - SWIR5)/(NIR + SWIR5)',{'NIR':image.select('B4'), 'SWIR5':image.select('B5')})
MNDWI = image.expression('(GREEN - SWIR5)/(GREEN + SWIR5)',{'GREEN':image.select('B2'), 'SWIR5':image.select('B5')})
EWI = image.expression('(GREEN - NIR - SWIR5)/(GREEN + NIR + SWIR5)',{'GREEN':image.select('B2'), 'NIR':image.select('B4'), 'SWIR5':image.select('B5')})
NEW = image.expression('(BLUE - SWIR7)/(BLUE + SWIR7)',{'BLUE':image.select('B1'), 'SWIR7':image.select('B7')})
NDWI_B = image.expression('(BLUE - NIR)/(BLUE+NIR)',{'BLUE':image.select('B1'), 'NIR':image.select('B4')})
AWElnsh = image.expression('4*(GREEN-SWIR5)-(0.25*NIR+2.75*SWIR7)',{'GREEN':image.select('B2'),'SWIR5':image.select('B5'),'NIR':image.select('B4'),'SWIR7':image.select('B7')})
# To get a collection of random points in a specified region, you can use: Create 1000 random points in the region.
randomPoints = ee.FeatureCollection.randomPoints(FCfromList);

# 第四步， 标注矢量数据转化为训练样本数据
https://github.com/pkuazi/gee_water.git

