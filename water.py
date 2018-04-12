#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-12 09:12
#

"""
qtp water extraction
""" 
import fiona
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
# randomPoints = ee.FeatureCollection.randomPoints(FCfromList);

# 第四步， 标注矢量数据转化为训练样本数据
# Collection query aborted after accumulating over 5000 elements样本点不能超过5000.能否直接读入已经处理好的training数据?
# training = image.sampleRegions(FCfromList)
# 直接从已有数据中生成training 数据
from imagepixel2trainingdata import Imagepixel_Trainingdata
import os
import pandas as pd

file_root = '/mnt/win/water_paper/training_data/TM'
training_dir = os.path.join(file_root, 'traing_csv')
image_bands_dir = os.path.join(file_root, 'image/L5134036')
sensor = 'TM'

imagepixel_trainingdata = Imagepixel_Trainingdata(file_root, image_bands_dir, sensor)
training_X, training_y = imagepixel_trainingdata.get_training_data()
# train_x, train_y, test_x, test_y = imagepixel_trainingdata.split_dataset(training_X,training_y, 0.1)
print('reading image data %s to be processed....' % imagepixel_trainingdata.image_bands_dir.split('/')[-1])
# 将csv record格式的训练数据转换为ee的training数据格式
whole_count, x_col = training_X.shape

training_y=training_y.rename(columns = {0:'landcover'})
whole_dt = pd.merge(training_X, training_y, left_index=True, right_index=True)

print(whole_dt.iloc[0])

# ['B10', 'B20', 'B30', 'B40', 'B50', 'B70']['NDWI', 'MNDWI', 'EWI', 'NEW', 'NDWI_B', 'AWElnsh']
trainFeat = []
for i in range(whole_count):
    trainpixel = ee.Feature(None, dict(whole_dt.iloc[i]))
    trainFeat.append(trainpixel)
trainFC = ee.FeatureCollection(trainFeat)

# 第五步 训练模型，并将指数特征波段加入到image中
# feature attributes list
train_bands = list(whole_dt.columns.values)
trained = ee.Classifier.cart().train(trainFC, 'landcover', train_bands)

# classify the image with the same bands used for training
newimage = image.addBands(NDWI, MNDWI, EWI, NEW, NDWI_B, AWElnsh)
classified = newimage.select(train_bands).classify(trained)

# 最后一步，结果导出
# 研究 https://developers.google.com/earth-engine/exporting

'''geometry:
null
properties:
Object (7 properties)
B2:
0.30924684
B3:
0.30622107
B4:
0.3287045
B5:
0.35653275
B6:
0.33084685
B7:
0.21383573
landcover:
0'''


# urban = ee.FeatureCollection(
#         [ee.Feature(
#             ee.Geometry.Point([-122.40898132324219, 37.78247386188714]),
#             {
#               "landcover": 0,
#               "system:index": "0"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.40623474121094, 37.77107659627034]),
#             {
#               "landcover": 0,
#               "system:index": "1"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.39799499511719, 37.785187237567705]),
#             {
#               "landcover": 0,
#               "system:index": "2"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.39936828613281, 37.772162125840445]),
#             {
#               "landcover": 0,
#               "system:index": "3"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.41104125976562, 37.76890548932033]),
#             {
#               "landcover": 0,
#               "system:index": "4"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.41859436035156, 37.7835592241132]),
#             {
#               "landcover": 0,
#               "system:index": "5"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.41790771484375, 37.801465399617314]),
#             {
#               "landcover": 0,
#               "system:index": "6"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.40142822265625, 37.77053382550901]),
#             {
#               "landcover": 0,
#               "system:index": "7"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.39662170410156, 37.75370595587201]),
#             {
#               "landcover": 0,
#               "system:index": "8"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.28950500488281, 37.8166551148543]),
#             {
#               "landcover": 0,
#               "system:index": "9"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.28195190429688, 37.82696064199382]),
#             {
#               "landcover": 0,
#               "system:index": "10"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.28126525878906, 37.81882481909333]),
#             {
#               "landcover": 0,
#               "system:index": "11"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.2723388671875, 37.82858769894982]),
#             {
#               "landcover": 0,
#               "system:index": "12"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.28607177734375, 37.84702517033112]),
#             {
#               "landcover": 0,
#               "system:index": "13"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.29293823242188, 37.8562421777618]),
#             {
#               "landcover": 0,
#               "system:index": "14"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.27302551269531, 37.849193981623955]),
#             {
#               "landcover": 0,
#               "system:index": "15"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.28057861328125, 37.86545803289311]),
#             {
#               "landcover": 0,
#               "system:index": "16"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.27096557617188, 37.820452055421086]),
#             {
#               "landcover": 0,
#               "system:index": "17"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.27920532226562, 37.808518155993234]),
#             {
#               "landcover": 0,
#               "system:index": "18"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.27783203125, 37.80092285199884]),
#             {
#               "landcover": 0,
#               "system:index": "19"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.31491088867188, 37.784644570400836]),
#             {
#               "landcover": 0,
#               "system:index": "20"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.31903076171875, 37.7835592241132]),
#             {
#               "landcover": 0,
#               "system:index": "21"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.26959228515625, 37.80200794325057]),
#             {
#               "landcover": 0,
#               "system:index": "22"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.40966796875, 37.768362702622596]),
#             {
#               "landcover": 0,
#               "system:index": "23"
#             })]),
# vegetation = ee.FeatureCollection(
#         [ee.Feature(
#             ee.Geometry.Point([-122.15835571289062, 37.81990964729775]),
#             {
#               "landcover": 1,
#               "system:index": "0"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.14462280273438, 37.806890656610484]),
#             {
#               "landcover": 1,
#               "system:index": "1"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.16522216796875, 37.817197546892785]),
#             {
#               "landcover": 1,
#               "system:index": "2"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.09793090820312, 37.80797566018445]),
#             {
#               "landcover": 1,
#               "system:index": "3"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.08282470703125, 37.81123057525427]),
#             {
#               "landcover": 1,
#               "system:index": "4"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.07733154296875, 37.7992951852321]),
#             {
#               "landcover": 1,
#               "system:index": "5"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.05398559570312, 37.77867496858311]),
#             {
#               "landcover": 1,
#               "system:index": "6"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.05398559570312, 37.76673431862507]),
#             {
#               "landcover": 1,
#               "system:index": "7"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.07389831542969, 37.792784159505125]),
#             {
#               "landcover": 1,
#               "system:index": "8"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.0306396484375, 37.83455326751277]),
#             {
#               "landcover": 1,
#               "system:index": "9"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.06497192382812, 37.831299380818606]),
#             {
#               "landcover": 1,
#               "system:index": "10"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.19268798828125, 37.85461573076714]),
#             {
#               "landcover": 1,
#               "system:index": "11"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.18170166015625, 37.849193981623955]),
#             {
#               "landcover": 1,
#               "system:index": "12"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.51609802246094, 37.84051835371829]),
#             {
#               "landcover": 1,
#               "system:index": "13"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.49137878417969, 37.838349287273296]),
#             {
#               "landcover": 1,
#               "system:index": "14"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.50511169433594, 37.82641828170282]),
#             {
#               "landcover": 1,
#               "system:index": "15"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.54081726074219, 37.84160286302103]),
#             {
#               "landcover": 1,
#               "system:index": "16"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.55592346191406, 37.85353141283498]),
#             {
#               "landcover": 1,
#               "system:index": "17"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.56278991699219, 37.86274760688767]),
#             {
#               "landcover": 1,
#               "system:index": "18"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.54631042480469, 37.86328970006369]),
#             {
#               "landcover": 1,
#               "system:index": "19"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.52708435058594, 37.85190490603355]),
#             {
#               "landcover": 1,
#               "system:index": "20"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.52845764160156, 37.83889155986444]),
#             {
#               "landcover": 1,
#               "system:index": "21"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.51472473144531, 37.83021472002989]),
#             {
#               "landcover": 1,
#               "system:index": "22"
#             })]),
# water = ee.FeatureCollection(
#         [ee.Feature(
#             ee.Geometry.Point([-122.61085510253906, 37.835095568009415]),
#             {
#               "landcover": 2,
#               "system:index": "0"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.60673522949219, 37.8166551148543]),
#             {
#               "landcover": 2,
#               "system:index": "1"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.60810852050781, 37.80038030039511]),
#             {
#               "landcover": 2,
#               "system:index": "2"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.57102966308594, 37.80472060163741]),
#             {
#               "landcover": 2,
#               "system:index": "3"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.58888244628906, 37.83455326751277]),
#             {
#               "landcover": 2,
#               "system:index": "4"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.65205383300781, 37.855157883752504]),
#             {
#               "landcover": 2,
#               "system:index": "5"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.62870788574219, 37.823164036248635]),
#             {
#               "landcover": 2,
#               "system:index": "6"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.62664794921875, 37.792784159505125]),
#             {
#               "landcover": 2,
#               "system:index": "7"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.60055541992188, 37.792784159505125]),
#             {
#               "landcover": 2,
#               "system:index": "8"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.53738403320312, 37.7992951852321]),
#             {
#               "landcover": 2,
#               "system:index": "9"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.57240295410156, 37.82641828170282]),
#             {
#               "landcover": 2,
#               "system:index": "10"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.58682250976562, 37.823164036248635]),
#             {
#               "landcover": 2,
#               "system:index": "11"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.36709594726562, 37.85570003275074]),
#             {
#               "landcover": 2,
#               "system:index": "12"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.40074157714844, 37.88171849539308]),
#             {
#               "landcover": 2,
#               "system:index": "13"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.40005493164062, 37.86925246182428]),
#             {
#               "landcover": 2,
#               "system:index": "14"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.37739562988281, 37.88117653780091]),
#             {
#               "landcover": 2,
#               "system:index": "15"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.34992980957031, 37.87358871277159]),
#             {
#               "landcover": 2,
#               "system:index": "16"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.34992980957031, 37.85244707895444]),
#             {
#               "landcover": 2,
#               "system:index": "17"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.34855651855469, 37.838349287273296]),
#             {
#               "landcover": 2,
#               "system:index": "18"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.39387512207031, 37.849193981623955]),
#             {
#               "landcover": 2,
#               "system:index": "19"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.44743347167969, 37.82262164805511]),
#             {
#               "landcover": 2,
#               "system:index": "20"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.43850708007812, 37.842687356377084]),
#             {
#               "landcover": 2,
#               "system:index": "21"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.354736328125, 37.789528431453014]),
#             {
#               "landcover": 2,
#               "system:index": "22"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.25826263427734, 37.68952589794135]),
#             {
#               "landcover": 2,
#               "system:index": "23"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.53326416015625, 37.81114015184751]),
#             {
#               "landcover": 2,
#               "system:index": "24"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.44503021240234, 37.87078823552829]),
#             {
#               "landcover": 2,
#               "system:index": "25"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.4532699584961, 37.86048883137166]),
#             {
#               "landcover": 2,
#               "system:index": "26"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.33036041259766, 37.833920567528345]),
#             {
#               "landcover": 2,
#               "system:index": "27"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.19989776611328, 37.65664491891396]),
#             {
#               "landcover": 2,
#               "system:index": "28"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.21946716308594, 37.62374937200642]),
#             {
#               "landcover": 2,
#               "system:index": "29"
#             }),
#         ee.Feature(
#             ee.Geometry.Point([-122.24040985107422, 37.61504728801728]),
#             {
#               "landcover": 2,
#               "system:index": "30"
#             })]);
#
# # // Merge the three geometry layers into a single FeatureCollection.
# newfc = urban.merge(vegetation).merge(water);
#
# # // Load the Landsat 8 scaled radiance image collection.
# landsatCollection = ee.ImageCollection('LANDSAT/LC08/C01/T1').filterDate('2017-01-01', '2017-12-31').filterMetadata('WRS_PATH', 'equals', 134).filterMetadata('WRS_ROW', 'equals', 36);
#
# # // Make a cloud-free composite.
# composite = ee.Algorithms.Landsat.simpleComposite({collection: landsatCollection,  asFloat: true});
#
# # // Use these bands for classification.
# bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];
# # // The name of the property on the points storing the class label.
# classProperty = 'landcover';
#
# # // Sample the composite to generate training data.  Note that the
# # // class label is stored in the 'landcover' property.
# training = composite.select(bands).sampleRegions({collection: newfc,properties: [classProperty], scale: 30});
#
#
# # // Train a CART classifier.
# classifier = ee.Classifier.cart().train({features: training, classProperty: classProperty,});
# # // Print some info about the classifier (specific to CART).
# print('CART, explained', classifier.explain());
#
# # // Classify the composite.
# classified = composite.classify(classifier);
# # Map.centerObject(newfc);
# # Map.addLayer(classified, {min: 0, max: 2, palette: ['red', 'green', 'blue']});
#
# # // Optionally, do some accuracy assessment.  Fist, add a column of
# # // random uniforms to the training dataset.
# withRandom = training.randomColumn('random');
#
# # // We want to reserve some of the data for testing, to avoid overfitting the model.
# split = 0.7;  # Roughly 70% training, 30% testing.
# trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
# testingPartition = withRandom.filter(ee.Filter.gte('random', split));
#
# # // Trained with 70% of our data.
# trainedClassifier = ee.Classifier.gmoMaxEnt().train({features: trainingPartition,classProperty: classProperty, inputProperties: bands});
#
# # // Classify the test FeatureCollection.
# test = testingPartition.classify(trainedClassifier);
#
# # // Print the confusion matrix.
# confusionMatrix = test.errorMatrix(classProperty, 'classification');
# print('Confusion Matrix', confusionMatrix);

