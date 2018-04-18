#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-18 11:25
#

"""
gee collection:
ImageCollection ID：https://explorer.earthengine.google.com/#detail/LANDSAT%2FLT05%2FC01%2FT1_SR
landsat5: 'LANDSAT/LT05/C01/T2'
landsat8: 'LANDSAT/LC08/C01/T1/'
The VNIR and SWIR bands have a resolution of 30m / pixel. The TIR band, while originally collected with a resolution of 120m / pixel (60m / pixel for Landsat 7) has been resampled using cubic-convolution to 30m.
GEE 进行了波段分辨率的统一（立方卷积进行插值）
""" 


# 统计分析GEE中qtp的landsat数据，及其数据质量