#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-11 09:08
#

"""

""" 
import rasterio
import os, re
import numpy as np
import rasterio.features
import fiona
import pandas as pd
from sklearn.utils import shuffle

def read_labeled_pixels(label_shp, tag, band_files, exception_band):
    '''
    :param label_shp: label polygons in shapefile format
    :param tag: the class of the polygon
    :param band_files: band files for this landsat image
    :param exception_band: some bands with different resolution
    :return: X,Y in DataFrame
    '''
    vector = fiona.open(label_shp, 'r')
    geom_list = []
    for feature in vector:
        # create a shapely geometry
        # this is done for the convenience for the .bounds property only
        # feature['geoemtry'] is in Json format
        geojson = feature['geometry']
        geom_list.append(geojson)

    X = np.array([])
    attributes = []
    # read each band pixels for the same geometries
    for band_file in band_files:
        print(band_file)
        band = re.findall(r'.*?_(B.*?).TIF', band_file)[0]

        # for band6, its resolution is different with other bands
        band_ok = True
        for e_band in exception_band:
            if e_band in band:
                band_ok = False
        if not band_ok:
            continue

        attributes.append(band)
        raster = rasterio.open(band_file, 'r')
        mask = rasterio.features.rasterize(geom_list, out_shape=raster.shape, transform=raster.transform, fill=0,
                                           all_touched=False, default_value=tag, dtype=np.uint8)

        data = raster.read(1)
        print(data.shape)
        assert mask.shape == data.shape
        # each band is a column in X
        X = np.append(X, data[mask == tag])
        # with rasterio.open("/tmp/mask.tif" , 'w', driver='GTiff', width=raster.width,height=raster.height, crs=raster.crs, transform=raster.transform, dtype=np.uint16,nodata=256,count=raster.count, indexes=raster.indexes) as dst:
        #     # Write the src array into indexed bands of the dataset. If `indexes` is a list, the src must be a 3D array of matching shape. If an int, the src must be a 2D array.
        #     dst.write(mask.astype(rasterio.uint16), indexes =1)

    # organize the Training samples in X and Y
    band_num = len(attributes)
    # # X has the same number of columns as the number of bands
    X = X.reshape(band_num, -1).T
    # # Y has the same rows as X, which is X.shape[1], column is X.shape[0]
    Y = np.repeat(tag, X.shape[0])

    X = pd.DataFrame(data=X, columns=attributes)
    Y = pd.DataFrame(data=Y, columns=['tag'])

    return X, Y


def generate_training_data(file_root, sensor):
    '''
    读取矢量标记的polygon数据，读取对应的栅格像素，注意矢量标记的命名规则：feat_name +‘-‘+影像L5+path+row
    :param file_root: folder of the label shapefiles and corresponding images
    :param sensor: landsat sensor: TM, ETM, OLI
    :return:save training pixels and labels into dataframe for each feature_image pair
    '''
    # for some bands, they have different resolution, so the samples number is not the same as others
    exception_band = {'TM': ['6'], 'ETM': ['6', '8'], 'OLI': ['8', '10', '11']}

    label_dir = os.path.join(file_root, 'feature_shp')
    img_dir = os.path.join(file_root, 'image')
    tags = {'water': 1, 'mount_shadow': 2, 'cloud': 3, 'cloud_shadow': 4, 'snow': 5, 'other': 6}

    # 建立存放训练数据csv文件的文件夹
    result_path = os.path.join(file_root, 'traing_csv')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for rst in os.listdir(img_dir):
        print('get training data samples for %s image' % rst)
        rst_dir = os.path.join(img_dir, rst)
        band_files = [os.path.join(rst_dir, name) for name in os.listdir(rst_dir) if
                      name.endswith('.tif') or name.endswith('.TIF')]

        # to find the corresponding label shapefiles according to the names, eg cloud-L5134036.shp
        for file in os.listdir(label_dir):
            if file.endswith(rst + ".shp"):
                print('the first label shapefile for this image is %s' % file)
                label_shp = os.path.join(label_dir, file)

                feat_name = re.findall(r'(.*?)-', file)[0]
                tag = tags[feat_name]

                print("generating the training data from the label shapefile")
                feat_x, feat_y = read_labeled_pixels(label_shp, tag, band_files, exception_band[sensor])
                # print(feat_name, feat_x.shape)
                result = pd.concat([feat_x, feat_y], axis=1)

                print('the attributes of the label from the images are %s' % feat_x.columns)

                result_name = os.path.join(result_path, '%s_%s.csv' % (rst, feat_name))
                result.to_csv(result_name)


# 读取转化为csv 记录的标注数据作为训练样本，添加特征属性，生成模型可以使用的数据集，同时读取待分类的遥感影像并保存为dataframe形式，并添加特征属性
class Imagepixel_Trainingdata:
    def __init__(self, file_root, image_bands_dir, sensor):
        self.file_root = file_root
        self.training_dir = os.path.join(file_root, 'traing_csv')
        self.image_bands_dir = image_bands_dir
        self.sensor = sensor

    def gen_features(self, original_X):
        '''
        using the six bands to generate other features, such as all kinds of water indeices
        :return: new traing X and Y
        '''
        band1 = original_X['B10']  # blue
        band2 = original_X['B20']  # green
        band3 = original_X['B30']  # red
        band4 = original_X['B40']  # nir
        band5 = original_X['B50']  # swir5
        band7 = original_X['B70']  # swir7
        NDWI = (band4 - band5) / (band4 + band5)
        MNDWI = (band2 - band5) / (band2 + band5)
        EWI = (band2 - band4 - band5) / (band2 + band4 + band5)
        NEW = (band1 - band7) / (band1 + band7)
        NDWI_B = (band1 - band4) / (band1 + band4)
        AWElnsh = 4 * (band2 - band5) - (0.25 * band4 + 2.75 * band7)

        features = pd.concat([NDWI, MNDWI, EWI, NEW, NDWI_B, AWElnsh], axis=1)
        features.columns = ['NDWI', 'MNDWI', 'EWI', 'NEW', 'NDWI_B', 'AWElnsh']
        # DataFrame({}data=[NDWI, MNDWI, EWI, NEW, NDWI_B, AWElnsh], columns=['NDWI', 'MNDWI', 'EWI', 'NEW', 'NDWI_B', 'AWElnsh'])
        # training_X = pd.merge(original_X, features, left_index=True, right_index=True)
        training_X = pd.concat([original_X, features], axis=1)

        return training_X

    def get_training_data(self):
        '''
        read the labeled pixels, and corresponding bands values
        :return: original training X, that is six bands values
        '''
        # training data, the features are the original six band values.
        for file in os.listdir(self.training_dir):
            if file.endswith('water.csv'):
                X_water = pd.read_csv(os.path.join(self.training_dir, file),
                                      usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('cloud.csv'):
                X_cloud = pd.read_csv(os.path.join(self.training_dir, file),
                                      usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('cloud_shadow.csv'):
                X_cloud_shadow = pd.read_csv(os.path.join(self.training_dir, file),
                                             usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('mount_shadow.csv'):
                X_mount_shadow = pd.read_csv(os.path.join(self.training_dir, file),
                                             usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('snow.csv'):
                X_snow = pd.read_csv(os.path.join(self.training_dir, file),
                                     usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('other.csv'):
                X_other = pd.read_csv(os.path.join(self.training_dir, file),
                                      usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
        X_nonwater = pd.concat([X_cloud, X_cloud_shadow, X_mount_shadow, X_snow, X_other])

        Y_water = pd.Series(np.repeat(1, X_water.shape[0]))
        Y_nonwater = pd.Series(np.repeat(0, X_nonwater.shape[0]))

        original_X = pd.concat([X_water, X_nonwater])
        training_Y = pd.concat([Y_water, Y_nonwater])

        training_X = self.gen_features(original_X)
        training_Y = pd.DataFrame(training_Y)
        training_Y.columns = ['landcover']
        return training_X, training_Y

    def split_dataset(self, x_df, y_df, frac):
        '''
        input x,y are dataframes, output are dataframes either, frac the percent of the training dataset, between 0 and 1, 0.8 for example
        :param x: x is a pd.DataFrame
        :param y: y is a pd.DataFrame
        :param frac: 训练样本的划分比例，介于0-1之间
        :return:
        '''
        whole_count, x_col = x_df.shape
        whole_dt = pd.merge(x_df, y_df, left_index=True, right_index=True)
        shuffle_dt = shuffle(whole_dt)
        shuffle_dt = shuffle_dt.reset_index(drop=True)

        # split dataset into training and testing
        split_line = int(whole_count * frac)
        x_train = shuffle_dt.ix[:split_line, :x_col]
        y_train = shuffle_dt.ix[:split_line, -1]
        x_test = shuffle_dt.ix[split_line + 1:, :x_col]
        y_test = shuffle_dt.ix[split_line + 1:, -1]

        return x_train, y_train, x_test, y_test


    def read_bands_to_be_classified(self):
        # for some bands, they have different resolution, so the samples number is not the same as others
        exception_band = {'TM': ['6'], 'ETM': ['6', '8'], 'OLI': ['8', '10', '11']}

        attributes = []
        values = []
        band_files = [os.path.join(self.image_bands_dir, name) for name in os.listdir(self.image_bands_dir) if
                      name.endswith('.tif') or name.endswith('.TIF')]
        for band_file in band_files:
            print(band_file)
            band = re.findall(r'.*?_(B.*?).TIF', band_file)[0]

            # for band6, its resolution is different with other bands
            band_ok = True
            for e_band in exception_band[self.sensor]:
                if e_band in band:
                    band_ok = False
            if not band_ok:
                continue

            attributes.append(band)

            raster = rasterio.open(band_file, 'r')
            # parameters will be used when storing array back into geotiff
            src_profile = raster.profile

            # reading test data using window ((row_start, row_stop), (col_start, col_stop))
            # array = raster.read(1,window=((1000,1010),(3000,3006)))
            array = raster.read(1)

            values.append(array.reshape(-1, 1))
        bands_num = len(attributes)
        value_array = np.array(values)
        resize_array = value_array.reshape(bands_num, -1)

        X = pd.DataFrame(resize_array.T, columns=attributes)
        print(X.shape)

        return X, src_profile

def test_generate_training_data():
    file_root = '/mnt/win/water_paper/training_data/TM'
    generate_training_data(file_root, 'TM')

def test():
    file_root = '/mnt/win/water_paper/training_data/TM'
    file_root = file_root
    image_bands_dir = os.path.join(file_root, 'image/L5134036')
    sensor = 'TM'

    imagepixel_trainingdata = Imagepixel_Trainingdata(file_root, image_bands_dir, sensor)
    training_X, training_y = imagepixel_trainingdata.get_training_data()

    # train_x, train_y, test_x, test_y = imagepixel_trainingdata.split_dataset(training_X, training_y, 0.1)



if __name__ == '__main__':
    # test_generate_training_data()
    test()
