#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-18 11:25
#

"""
gee collection:
ImageCollection ID：https://explorer.earthengine.google.com/#detail/LANDSAT%2FLT05%2FC01%2FT1_SR
landsat5: 'LANDSAT/LT05/C01/T2' (USGS Landsat 5 Collection 1 Tier 2 Raw Scenes)
landsat4: 'LANDSAT/LT04/C01/T1' (USGS Landsat 4 Collection 1 Tier 1 Raw Scenes)
landsat8: 'LANDSAT/LC08/C01/T1/'
The VNIR and SWIR bands have a resolution of 30m / pixel. The TIR band, while originally collected with a resolution of 120m / pixel (60m / pixel for Landsat 7) has been resampled using cubic-convolution to 30m.
GEE 进行了波段分辨率的统一（立方卷积进行插值）
"""

# 统计分析GEE中qtp的landsat数据，及其数据质量
import ee

ee.Initialize()

# // imoprt qtp boundary as a ee.Feature
bound = ee.FeatureCollection('users/jianghua512/qtp_boundary_wgs84')
print(bound.getInfo())
# Map.addLayer(bound, {color: '0000FF'}, 'Qinghai-Tibetan Plateau');

bound_geom = bound.getInfo()['features'][0]['geometry']
print(bound_geom);

# // Load Landsat 5 data, filter by dates and bounds.
# l5_collection = ee.ImageCollection('LANDSAT/LT05/C01/T2').filterDate('1987-01-01','1987-12-31').filterBounds(bound_geom);
l5_collection = ee.ImageCollection('LANDSAT/LT05/C01/T2').filterBounds(bound_geom)
# // Convert the collection to a list and get the number of images.
size = l5_collection.toList(100000).length()
size = size.getInfo()
print('Number of images: ', size)  # 12598

# // Get the number of images.
count = l5_collection.size()
count = count.getInfo()
print('Count: ', count)

# // Get the date range of images in the collection.
dates = ee.List(l5_collection.get('date_range'))
dateRange = ee.DateRange(dates.get(0), dates.get(1))
dateRange = dateRange.getInfo()
print('Date range: ', dateRange)  #

# // Get statistics for a property of the images in the collection.
sunStats = l5_collection.aggregate_stats('SUN_ELEVATION');
print('Sun elevation statistics: ', sunStats);

cloudCover = l5_collection.aggregate_stats('CLOUD_COVER').getInfo()
'''
Aggregates over a given property of the objects in a collection, calculating the
sum, min, max, mean, sample standard deviation, sample variance, total standard deviation and total variance of the selected property.
max: 100
mean: 75.43951420860454
min: -1
sample_sd: 29.319729461053385
sample_var: 859.6465356693617
sum: 950387
sum_sq: 82525701
total_count: 12598
total_sd: 29.318565771923474
total_var: 859.5782989226027
valid_count: 12598
weight_sum: 12598
weighted_sum: 950387
'''
# # // Sort by a cloud cover property, get the least cloudy image.
# image = ee.Image(l5_collection.sort('CLOUD_COVER').first());
# print('Least cloudy image: ', image);
#
# # // Limit the collection to the 10 most recent images.
# recent = l5_collection.sort('system:time_start', False).limit(10);
# print('Recent images: ', recent);
#
# # // Load Landsat 5 data, filter by dates and bounds.
# l5_collection = ee.ImageCollection('LANDSAT/LT05/C01/T2').filterDate('1987-01-01', '1987-12-31').filterBounds(
#     bound_geom);
# print(l5_collection.getInfo());

# // Load Landsat 8 data, filter by bounds.
l8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1').filterBounds(bound_geom)
cloudCover = l8_collection.aggregate_stats('CLOUD_COVER').getInfo()
print(l8_collection.getInfo())
'''
max: 100
mean: 31.366992586785614
min: 0
sample_sd: 28.046576907788168
sample_var: 786.6104762444766
sum: 517116.2397857476
sum_sq: 29187654.960756194
total_count: 16486
total_sd: 28.04572627688226
total_var: 786.562762397804
valid_count:16486
weight_sum: 16486
weighted_sum: 517116.2397857476
'''

# // Load a Landsat 8 ImageCollection for a single path-row.
collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
.filter(ee.Filter.eq('WRS_PATH', 44))
.filter(ee.Filter.eq('WRS_ROW', 34))
.filterDate('2014-01-01', '2015-01-01');
print('Collection: ', collection);

# // Convert the collection to a list and get the number of images.
size = collection.toList(100).length();
print('Number of images: ', size);

# // Get the number of images.
count = collection.size();
print('Count: ', count);

# // Get the date range of images in the collection.
dates = ee.List(collection.get('date_range'));
dateRange = ee.DateRange(dates.get(0), dates.get(1));
print('Date range: ', dateRange);

# // Get statistics for a property of the images in the collection.
sunStats = collection.aggregate_stats('SUN_ELEVATION');
print('Sun elevation statistics: ', sunStats);

# // Sort by a cloud cover property, get the least cloudy image.
image = ee.Image(collection.sort('CLOUD_COVER').first());
print('Least cloudy image: ', image);

# // Limit the collection to the 10 most recent images.
recent = collection.sort('system:time_start', false).limit(10);
print('Recent images: ', recent);
