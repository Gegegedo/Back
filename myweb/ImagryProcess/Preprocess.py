import os,shutil
import numpy as np
import re
import gdal
from myweb.models import Bmap
from myweb.models import Mask
from myweb.Detector import detection
from django.contrib.gis.geos import LinearRing,Polygon,MultiPolygon
import cv2
from collections import defaultdict
import requests
import time
from geoserver.catalog import Catalog
from django.contrib.gis.gdal import CoordTransform,SpatialReference
MAPBASEPATH='/media/zhou/Document/yaogan/TJ'
mask_url='http://localhost:8080/geoserver/rest/workspaces/Mask/datastores/Mask/featuretypes'
map_url="http://localhost:8080/geoserver/rest/"
def preprogress(id):
    def getUploadFile(uploadfolder):
        try:
            uploadfiles = os.listdir(uploadfolder)  # 上传的文件夹
            while(len(uploadfiles)==1):
                uploadfolder=os.path.join(uploadfolder,uploadfiles[0])
                uploadfiles=os.listdir(uploadfolder)
            for file in uploadfiles:
                if(file[-11:]=='_fusion.tif'):#融合图
                    fusionname=file
                    capture_time=re.match('[\w\_\.]+_(\d{8})_',fusionname).group(1)
                    Bmap.objects.filter(id=id).update(capture_time=time.strftime('%Y-%m-%d',time.strptime(capture_time,'%Y%m%d')))
                # elif re.search(r'PAN\d.jpg',file):#缩略图
                #     thumbnailname=file
                # elif re.search(r'PAN\d.xml',file):#XML
                #     XMLname=file
                elif re.search(r'_rpc.txt',file):#RPC
                    rpcfile=file
                # elif re.search(r'MSS\d.rpb',file):
                #     rpbfile=file
            return uploadfolder,fusionname,rpcfile
        except Exception as e:
            return Exception("上传失败，请检查地图名称!")

    ####转8比特三通道,生成缩略图
    def chaneltransform():
        try:
            fusionimage=os.path.join(uploadfiles[0],uploadfiles[1])
            gdal.AllRegister()
            driver = gdal.GetDriverByName("GTiff")
            fusionimage = gdal.Open(fusionimage.encode('utf-8').decode(), gdal.GA_ReadOnly)
            im_width = fusionimage.RasterXSize
            im_height = fusionimage.RasterYSize
            transformimage = os.path.join(uploadfiles[0],"chaneltransform.tif")
            dstDS = driver.Create(transformimage,
                                  xsize=im_width, ysize=im_height, bands=3, eType=gdal.GDT_Byte)
            thumbnail=np.zeros(shape=(int(im_height*0.02),int(im_width*0.02),3))
            for iband in range(1, 4):
                imgMatrix = fusionimage.GetRasterBand(iband).ReadAsArray(0, 0, im_width, im_height)
                zeros = np.size(imgMatrix) - np.count_nonzero(imgMatrix)
                minVal = np.percentile(imgMatrix, float(zeros / np.size(imgMatrix) * 100 + 0.15))
                maxVal = np.percentile(imgMatrix, 99)

                idx1 = imgMatrix < minVal
                idx2 = imgMatrix > maxVal
                idx3 = ~idx1 & ~idx2
                imgMatrix[idx1] = imgMatrix[idx1] * 20 / minVal
                imgMatrix[idx2] = 255
                idx1=None
                idx2=None
                imgMatrix[idx3] = pow((imgMatrix[idx3] - minVal) / (maxVal - minVal), 0.9) * 255
                if iband==1:
                    dstDS.GetRasterBand(3).WriteArray(imgMatrix)
                    dstDS.FlushCache()
                    thumbnail[:, :, 2] = cv2.resize(src=imgMatrix, dsize=(thumbnail.shape[1], thumbnail.shape[0]))
                    imgMatrix = None
                elif iband==2:
                    dstDS.GetRasterBand(2).WriteArray(imgMatrix)
                    dstDS.FlushCache()
                    thumbnail[:, :, 1] = cv2.resize(src=imgMatrix, dsize=(thumbnail.shape[1], thumbnail.shape[0]))
                    imgMatrix = None
                else:
                    dstDS.GetRasterBand(1).WriteArray(imgMatrix)
                    dstDS.FlushCache()
                    thumbnail[:, :, 0] = cv2.resize(src=imgMatrix, dsize=(thumbnail.shape[1], thumbnail.shape[0]))
                    imgMatrix = None
            fusionimage = None
            dstDS = None
            cv2.imwrite(os.path.join(uploadfiles[0],str(id)+'.jpg'),thumbnail)
            return transformimage
        except Exception as e:
            return Exception("上传失败，图像转换出错:"+str(e))

    def RPCOrthorectification(Alpha=True,is_label=False):
        try:
            if not is_label:
                orginalimage=os.path.join(uploadfiles[0],'chaneltransform.tif')
                transform_rpc = os.path.join(uploadfiles[0], 'chaneltransform_rpc.txt')
            else:
                orginalimage=os.path.join(uploadfiles[0],'label.tif')
                transform_rpc=os.path.join(uploadfiles[0],'label_rpc.txt')
            origin_rpc=os.path.join(uploadfiles[0],uploadfiles[2])
            shutil.copyfile(origin_rpc,transform_rpc)
            # with open(rpbfile,'r') as f:
            #     for line in f.readlines():
            #         hoffLine=re.search(r'heightOffset = ([\+|\-|\d]\d+\.?\d+)',line)
            #         if hoffLine:
            #             hoff=hoffLine.group(1)
            #             break
            # f.close()
            # RpcHeight="['RPC_HEIGHT="+str(hoff)+"]'"
            # transformerOptions=RpcHeight

            if Alpha:
                warpOP = gdal.WarpOptions(dstSRS='WGS84', rpc=True, multithread=True, errorThreshold=0.0,creationOptions=['Tiled=yes'],
                                      resampleAlg=gdal.gdalconst.GRIORA_Bilinear,dstAlpha=True)
            else:
                warpOP = gdal.WarpOptions(dstSRS='WGS84', rpc=True, multithread=True, errorThreshold=0.0,creationOptions=['Tiled=yes'],
                                          resampleAlg=gdal.gdalconst.GRIORA_Bilinear,dstNodata=0)
            image = gdal.Open(orginalimage.encode('utf-8'),gdal.GA_ReadOnly)
            RPCOrthImage = os.path.join(uploadfiles[0],os.path.basename(orginalimage).replace(".tif","RPC.tif"))
            srcDS = gdal.Warp(RPCOrthImage.encode('utf-8').decode(), image, options=warpOP)
            image=None
            srcDS=None
            return RPCOrthImage
        except Exception as e:
            return Exception("上传失败，RPC正射校正出错:"+str(e))

    def buildOverviews():
        try:
            image=os.path.join(uploadfiles[0],'chaneltransformRPC.tif')
            gdal.AllRegister()
            TransformDS = gdal.Open(image.encode('utf-8').decode(), gdal.GA_ReadOnly)
            Width = TransformDS.RasterXSize
            Heigh = TransformDS.RasterYSize
            PixelNum = Width * Heigh
            TopNum = 4096
            CurNum = PixelNum / 4
            anLevels = []
            nLevelCount = 0
            while (CurNum > TopNum):
                anLevels.append(pow(2, nLevelCount + 2))
                nLevelCount += 1
                CurNum /= 4
            TransformDS.BuildOverviews(overviewlist=anLevels)
            cat = Catalog(map_url,'admin', 'geoserver')
            wkspce = cat.get_workspace('Map')
            cat.create_coveragestore_external_geotiff(name=id, data='file://' + image.encode('utf-8').decode('utf-8'),
                                                      workspace=wkspce)
            cat.reload()
            TransformDS = None
        except Exception as e:
            return Exception("上传失败，建立金字塔出错"+str(e))

    def fit_by_contours(img,geotransfrom):
        geo = np.array([[geotransfrom[1], geotransfrom[4]], [geotransfrom[2], geotransfrom[5]]])
        off = np.array([geotransfrom[0], geotransfrom[3]])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[1:]
        contours = list(map(np.squeeze, contours))
        hierarchy = np.squeeze(hierarchy)
        contours = [LinearRing(np.dot(np.row_stack((single_contour, single_contour[0, :])), geo) + off) for
                    single_contour in contours]
        hole_exclude_linering = defaultdict(list)
        for idx in np.argwhere(hierarchy[:, -1] == -1)[:, 0]:
            hole_exclude_linering[idx].append(contours[idx])
        # external_contours=[(idx,contours[idx]) for idx in np.argwhere(hierarchy[:,-1]==-1)[:,0]]
        extern_linering_idx = np.argwhere(hierarchy[:, 2] != -1)[:, 0]
        hole_idx = [np.argwhere(hierarchy[:, -1] == idx)[:, 0] for idx in extern_linering_idx]
        for e_id, h_id in zip(extern_linering_idx, hole_idx):
            holes = [contours[h] for h in h_id]
            hole_exclude_linering[e_id].extend(holes)
        return MultiPolygon([Polygon(*linering) for linering in hole_exclude_linering.values()])

    def save_mask():
        try:
            ct=CoordTransform(SpatialReference('WGS84'), SpatialReference('4527'))
            label_path=os.path.join(uploadfiles[0],'labelRPC.tif')
            dataset = gdal.Open(label_path)
            GeoTransform = dataset.GetGeoTransform()
            if dataset == None:
                return
            im_width = dataset.RasterXSize  # 栅格矩阵的列数
            im_height = dataset.RasterYSize  # 栅格矩阵的行数
            cood_trans=lambda L,C:(GeoTransform[0] + C * GeoTransform[1] + L * GeoTransform[2],GeoTransform[3] + C * GeoTransform[4] + L * GeoTransform[5])
            map_polygon=Polygon(LinearRing(cood_trans(0,0),cood_trans(0,im_width),cood_trans(im_height,im_width),cood_trans(im_height,0),cood_trans(0,0)))
            Bmap.objects.filter(id=id).update(polygon=map_polygon)
            im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
            dataset = None
            types = np.unique(im_data)
            for label_type in types:
                # if label_type in (0,):
                #     continue
                mp = fit_by_contours((im_data == label_type).astype(np.uint8), GeoTransform)
                m = Mask(map=Bmap.objects.get(id=id),type_id=int(label_type), mask=mp,area=mp.transform(ct).area)
                m.save()
                # img[im_data == label_type]=127
                # cv2.imwrite(str(label_type)+".jpg",img)
                if label_type!=0:
                    payload = "<featureType><name>" + str(id) + '_' + str(m.type_id) + "</name><nativeName>myweb_mask</nativeName>"" \
                    ""<cqlFilter>type_id=" + str(m.type_id) + " and map_id=" + str(id) + "</cqlFilter></featureType>"
                    headers = {'Content-type': 'text/xml'}
                    resp = requests.post(mask_url, auth=('admin', 'geoserver'), data=payload, headers=headers)
                    if resp.status_code != 201:
                        raise Exception('Upload to geoserver error')
                    else:
                        cat = Catalog(map_url, 'admin', 'geoserver')
                        layer = cat.get_layer('Mask:'+str(id)+'_'+str(m.type_id))
                        layer.default_style=cat.get_style(str(label_type), 'Mask')
                        cat.save(layer)
                        cat.reload()
            return "上传成功"
        except Exception as e:
            return Exception("上传失败,拟合图斑出错:"+str(e))
#
#     def makeDownload():
#         try:
#             cwd = os.getcwd()
#             downloadpath=os.path.join(baseurl,str(id)+'.tar.gz')#前端下载路径
#             downloadfile = tarfile.open(downloadpath, "w:gz")
#             os.chdir(uploadfiles[0])
#             downloadfile.add(uploadfiles[4],recursive=False)
#             downloadfile.add(uploadfiles[5],recursive=False)
#             os.chdir(tempfolder)
#             for file in os.listdir(tempfolder):
#                 if ".json" in file:
#                     downloadfile.add(file,recursive=False)
#                 if file=="chaneltransformRPC.tif":
#                     downloadfile.add(file)
#             downloadfile.close()
#             os.chdir(cwd)
#             return downloadpath
#         except Exception:
#             # if os.path.exists(downloadpath):
#             #     os.remove(downloadpath)
#             return "上传失败，无法创建压缩包"
#     if not os.path.exists(baseurl):
#         os.makedirs(baseurl)


    # uploadfiles=(os.path.join(MAPBASEPATH,'GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826'),)
    # result = detection.detection(uploadfiles[0] + "/", uploadfiles[1], uploadfiles[0])
    # if not isinstance(result, Exception):
    # result =save_mask()
    # if not isinstance(result, Exception):
    #     result = save_mask()
    #     if not isinstance(result, Exception):
    #         # result=makeDownload()
    #             return '上传成功'
    # buildOverviews()
    # RPCOrthorectification()
    #正式代码
    # uploadfolder=os.path.join(MAPBASEPATH,'GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826')
    # uploadfiles=getUploadFile(uploadfolder)
    # RPCOrthorectification(Alpha=False,is_label=True)
    uploadfolder =os.path.join(MAPBASEPATH,Bmap.objects.get(id=id).name)
    uploadfiles=getUploadFile(uploadfolder)
    result = uploadfiles
    if not isinstance(result, Exception):
        result=chaneltransform()
        if not isinstance(result,Exception):
            result=RPCOrthorectification()
            if not isinstance(result,Exception):
                result=buildOverviews()
                if not isinstance(result,Exception):
                    result=detection.detection(uploadfiles[0]+"/",uploadfiles[1],uploadfiles[0])
                    if not isinstance(result,Exception):
                        result=RPCOrthorectification(Alpha=False,is_label=True)
                        if not isinstance(result,Exception):
                            result=save_mask()
                            if not isinstance(result,Exception):
                                shutil.rmtree('./myweb/Detector/temp')
                                return '上传成功'

    ####正式代码完
    cat = Catalog(map_url, 'admin', 'geoserver')
    if cat.get_layer('Mask:'+str(id)):
        cat.delete(cat.get_layer('Mask:'+str(id)))
        cat.reload()
    for label_type in range(1,8):
        if cat.get_layer('Mask:' + str(id)+'_'+str(label_type)):
            cat.delete(cat.get_layer('Mask:' + str(id)+'_'+str(label_type)))
            cat.reload()
    try:
        if cat.get_store(name=str(id), workspace='Map'):
            cat.delete(cat.get_store(name=str(id), workspace='Map'))
            cat.reload()
    except Exception:
        pass
    if os.path.exists('./myweb/Detector/temp'):
        shutil.rmtree('./myweb/Detector/temp')
    Bmap.objects.filter(id=id).delete()
    return str(result)