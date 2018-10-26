import json
import os
from itertools import chain
import gdal
import numpy as np
from myweb.models import Mask
import requests
import cv2
from django.contrib.gis.geos import LinearRing,Polygon,MultiPolygon
from collections import defaultdict
def geojson_make(label_path, save_path):

    app=6.4e-07
    def transforms(x,y):
        # pass
        px = GeoTransform[0] + x * GeoTransform[1] + y * GeoTransform[2]
        py = GeoTransform[3] + x * GeoTransform[4] + y * GeoTransform[5]
        # print(px,py)
        # [ppy,ppx]=WCS_transform.wgs2mercator(py,px)
        # return [ppx,ppy]
        return [px,py]
    try:
        dataset = gdal.Open(label_path)
        GeoTransform=dataset.GetGeoTransform()
        if dataset == None:
            return
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        dataset=None
        types = np.unique(im_data)

        # types=np.array([0]).astype('uint8')
        label_area = {}
        for label_type in types:
            if label_type==0:
                continue
            geojsons = []
            img = (im_data == label_type).astype('int8')
            pfit = fit(img)
            pfit.poly_fit()
            label_area[int(label_type)] = 0
            for poly in pfit.polys:
                segmentation = list(chain.from_iterable(zip(poly.boundary.xy[1], poly.boundary.xy[0])))
                geojson = dict()
                geometry = dict()
                geometry['coordinates'] = [[transforms(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]]
                geometry['type'] = "Polygon"
                geojson['geometry'] = geometry
                geojson['properties'] = {'type': str(label_type)}
                geojson['type'] = 'Feature'
                geojsons.append(geojson)
                label_area[label_type] = label_area[label_type]+poly.area
            with open(os.path.join(save_path, 'geojsons_t' + str(label_type)+'.json'), 'w') as json_file:
                json.dump(geojsons, json_file, ensure_ascii=False)
        # label_area={0:12,1:56,3:45,4:89}
        with open(os.path.join(save_path, 'label_area.json'), 'w') as json_file:
            label_area_dict=[]
            for key,value in label_area.items():
                label_area_dict.append({key:value*app})
            json.dump(label_area_dict, json_file, ensure_ascii=False)
    except Exception as e:
        return e

def fit_by_contours(img,geotransfrom):
    geo=np.array([[geotransfrom[1],geotransfrom[4]],[geotransfrom[2],geotransfrom[5]]])
    off=np.array([geotransfrom[0],geotransfrom[3]])
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    contours,hierarchy=cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[1:]
    contours=list(map(np.squeeze,contours))
    hierarchy=np.squeeze(hierarchy)
    contours=[LinearRing(np.dot(np.row_stack((single_contour, single_contour[0, :])), geo) + off) for single_contour in contours]
    hole_exclude_linering=defaultdict(list)
    for idx in np.argwhere(hierarchy[:,-1]==-1)[:,0]:
        hole_exclude_linering[idx].append(contours[idx])
    # external_contours=[(idx,contours[idx]) for idx in np.argwhere(hierarchy[:,-1]==-1)[:,0]]
    extern_linering_idx=np.argwhere(hierarchy[:,2]!=-1)[:,0]
    hole_idx=[np.argwhere(hierarchy[:,-1]==idx)[:,0] for idx in extern_linering_idx]
    for e_id,h_id in zip(extern_linering_idx,hole_idx):
        holes=[contours[h] for h in h_id]
        hole_exclude_linering[e_id].extend(holes)
    return MultiPolygon([Polygon(*linering) for linering in hole_exclude_linering.values()])
def save_polygon(label_path):
    try:
        dataset = gdal.Open(label_path)
        GeoTransform = dataset.GetGeoTransform()
        if dataset == None:
            return
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        dataset = None
        types = np.unique(im_data)
        for label_type in types:
            if label_type in (0,):
                continue
            mp=fit_by_contours((im_data == label_type).astype(np.uint8),GeoTransform)
            m = Mask(map_name='Test', type_id=int(label_type), mask=mp)
            m.save()
            # img[im_data == label_type]=127
            # cv2.imwrite(str(label_type)+".jpg",img)
            myUrl = 'http://localhost:8080/geoserver/rest/workspaces/GF2/datastores/Back/featuretypes'
            payload = "<featureType><name>"+m.map_name+'_'+str(m.type_id)+"</name><nativeName>myweb_mask</nativeName>" \
                                                                          "<cqlFilter>type_id="+str(m.type_id)+" and map_name="+'\''+m.map_name+"\'</cqlFilter></featureType>"
            headers = {'Content-type': 'text/xml'}
            resp = requests.post(myUrl, auth=('admin', 'geoserver'), data=payload, headers=headers)
            if resp.status_code!=201:
               raise Exception('Upload to geoserver error')
        return "上传成功"
    except Exception as e:
        if Mask.objects.filter(map_name='Test', type_id=int(label_type)):
            Mask.objects.filter(map_name='Test',type_id=int(label_type)).delete()
        return str(e)+":上传失败"
def main():
    save_polygon('/home/zhou/Back/fusion.tif')
    return 0

if __name__ == '__main__':
    main()
