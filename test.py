from django.contrib.gis.gdal import GDALRaster
rast=GDALRaster('fusionRPC.tif')
print (rast.extent)