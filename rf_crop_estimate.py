from osgeo import gdal, gdal_array,osr
from osgeo import ogr
from osgeo import gdal
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import matplotlib.pyplot as plt

gdal.UseExceptions()
gdal.AllRegister()

#File with bounding boxes of all classes samples
samples_shp_file = r"E:\NotebookStart\random_forest\Samples_compiled\samples_rf_new.shp"

#File to be used for samples extraction
raster_for_samples = r"E:\NotebookStart\random_forest\After_Land_Water_final_sat_bfsml.tif"

#File Where samples chips to be save
samples_chips = r"E:\NotebookStart\samples_chips.tif"

def samples_chips_extract(samples_shp_file,raster_for_samples,samples_chips):
    # Open the dataset from the file
    dataset = ogr.Open(samples_shp_file)
    # Make sure the dataset exists -- it would be None if we couldn't open it
    if not dataset:
        print('Error: could not open dataset')
    #######################################################################################################################
    ### Let's get the driver from this file
    driver = dataset.GetDriver()
    print('Dataset driver is: {n}\n'.format(n=driver.name))

    ### How many layers are contained in this Shapefile?
    layer_count = dataset.GetLayerCount()
    print('The shapefile has {n} layer(s)\n'.format(n=layer_count))

    ### What is the name of the 1 layer?
    layer = dataset.GetLayerByIndex(0)
    print('The layer is named: {n}\n'.format(n=layer.GetName()))

    ### What is the layer's geometry? is it a point? a polyline? a polygon?
    # First read in the geometry - but this is the enumerated type's value
    geometry = layer.GetGeomType()

    # So we need to translate it to the name of the enum
    geometry_name = ogr.GeometryTypeToName(geometry)
    print("The layer's geometry is: {geom}\n".format(geom=geometry_name))

    ### What is the layer's projection?
    # Get the spatial reference
    spatial_ref = layer.GetSpatialRef()

    # Export this spatial reference to something we can read... like the Proj4
    proj4 = spatial_ref.ExportToProj4()
    print('Layer projection is: {proj4}\n'.format(proj4=proj4))

    ### How many features are in the layer?
    feature_count = layer.GetFeatureCount()
    print('Layer has {n} features\n'.format(n=feature_count))

    ### How many fields are in the shapefile, and what are their names?
    # First we need to capture the layer definition
    defn = layer.GetLayerDefn()

    # How many fields
    field_count = defn.GetFieldCount()
    print('Layer has {n} fields'.format(n=field_count))

    # What are their names?
    print('Their names are: ')
    for i in range(field_count):
        field_defn = defn.GetFieldDefn(i)
        print('\t{name} - {datatype}'.format(name=field_defn.GetName(),
                                            datatype=field_defn.GetTypeName()))

    #######################################################################################################################

    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(raster_for_samples, gdal.GA_ReadOnly)

    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(samples_chips, ncol, nrow, 1, gdal.GDT_Byte)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)

    # Rasterize the shapefile layer to our new dataset
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                [1],  # output to our new dataset's first band
                                layer,  # rasterize this layer
                                None, None,  # don't worry about transformations since we're in same projection
                                [0],  # burn value 0
                                ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                'ATTRIBUTE=id']  # put raster values according to the 'id' field values
                                )

    # Close dataset
    out_raster_ds = None

    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")

def pixels_class(samples_chips):
    roi_ds = gdal.Open(samples_chips, gdal.GA_ReadOnly)
    roi = roi_ds.GetRasterBand(1).ReadAsArray()
    # How many pixels are in each class?
    classes = np.unique(roi)
    # Iterate over all class labels in the ROI image, printing out some information
    for c in classes:
        print('Class {c} contains {n} pixels'.format(c=c, n=(roi == c).sum()))

#######################################################################################################################
def GetGeoInfo(FileName):
    if os.path.exists(FileName) is False:
        raise Exception('[Errno 2] No such file or directory: \'' + FileName + '\'')

    SourceDS = gdal.Open(FileName, gdal.GA_ReadOnly)
    if SourceDS == None:
        raise Exception("Unable to read the data file")

    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = SourceDS.GetRasterBand(1).DataType
    DataType = gdal.GetDataTypeName(DataType)

    return NDV, xsize, ysize, GeoT, Projection, DataType
#######################################################################################################################

# samples_chips = r"E:\NotebookStart\samples_chips.tif"
output_image_for_detection = r"E:\NotebookStart\random_forest\After_Land_Water_final_sat_bfsml.tif"

#File to store detected output
output_detected = r"E:\NotebookStart\result_RF.tif"

#band numbers, for monitoring importance in training
# bands_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19]

def rf_detect(samples_chips, output_image_for_detection, output_detected):
    # Read in our image and ROI image
    img_ds = gdal.Open(output_image_for_detection, gdal.GA_ReadOnly)
    roi_ds = gdal.Open(samples_chips, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    bands_list = []
    for b in range(img.shape[2]):
        bands_list.append(b+1)

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    # Find how many non-zero entries we have -- i.e. how many training data samples?
    n_samples = (roi > 0).sum()
    print('We have {n} samples'.format(n=n_samples))
    # What are our classification labels?
    labels = np.unique(roi[roi > 0])
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))
    # We will need a "X" matrix containing our features, and a "y" array containing our labels
    #     These will have n_samples rows
    #     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster

    X = img[roi > 0, :]  # include all bands band, which is Fmask, for now
    y = roi[roi > 0]

    print('Our X matrix is sized: {sz}'.format(sz=X.shape))
    print('Our y array is sized: {sz}'.format(sz=y.shape))

    # clear = X[:, 18] <= 1
    #
    # X = X[clear, :18]  # we can ditch the Fmask band now
    # y = y[clear]

    print('After masking, our X matrix is sized: {sz}'.format(sz=X.shape))
    print('After masking, our y array is sized: {sz}'.format(sz=y.shape))

    # Initialize our model with 500 trees
    
    rf = RandomForestClassifier(n_estimators=500, oob_score=True)

    # Fit our model to training data
    rf = rf.fit(X, y)

    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))

    bands = bands_list

    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))

    # Setup a dataframe -- just like R
    df = pd.DataFrame()
    df['truth'] = y
    df['predict'] = rf.predict(X)

    # Cross-tabulate predictions
    print(pd.crosstab(df['truth'], df['predict'], margins=True))

    #########################################################################################################################
    # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])

    img_as_array = img[:, :, :19].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape,
                                            n=img_as_array.shape))

    # Now predict for each pixel
    class_prediction = rf.predict(img_as_array)

    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)

    print(type(class_prediction))

    ncol = img_ds.RasterXSize
    nrow = img_ds.RasterYSize

    NDV,xsize,ysize,GeoT,srs,DataType = GetGeoInfo(output_image_for_detection)
    ncol, nrow = xsize,ysize
    memory_driver = gdal.GetDriverByName('GTiff')
    result_raster = memory_driver.Create(output_detected, ncol, nrow, 1, gdal.GDT_Byte)
    result_raster.SetGeoTransform(GeoT)
    result_raster.SetProjection(srs.ExportToWkt())
    result_raster.GetRasterBand(1).WriteArray(class_prediction)


    # # Display them
    # plt.subplot(121)
    # plt.imshow(img[:, :, 4], cmap=plt.cm.Greys_r)
    # plt.title('SWIR1')
    #
    # plt.subplot(122)
    # plt.imshow(roi, cmap=plt.cm.Spectral)
    # plt.title('ROI Training Data')
    #
    # plt.show()

samples_chips_extract(samples_shp_file,raster_for_samples,samples_chips)

pixels_class(samples_chips)

rf_detect(samples_chips, output_image_for_detection, output_detected)
