import gdal
import numpy as np
import cv2
import os
from collections import defaultdict
def normalize_tif(img):
    max_pixel=np.max(img)
    normalize_img=img//max_pixel*256
    return normalize_img

def read_tif(MS_path,PAN_path):
    MS_dataset=gdal.Open(MS_path)
    PAN_dataset=gdal.Open(PAN_path)
    MS=MS_dataset.ReadAsArray(0,0,MS_dataset.RasterXSize,MS_dataset.RasterYSize)
    PAN=PAN_dataset.ReadAsArray(0,0,PAN_dataset.RasterXSize,PAN_dataset.RasterYSize)
    MS_normalize=MS/MS.max()*255
    PAN_normalize=(PAN/PAN.max()*255).astype(np.uint8)
    fusion_MS = np.mean(MS_normalize, axis=0).astype(np.uint8)
    # LRPAN = cv2.resize(PAN_normalize, dsize=(fusion_MS.shape[1], fusion_MS.shape[0]), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('',fusion_MS)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('',LRPAN)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # write_tif(fusion_MS,'F:\GF2_Registration','onebandMS.tif')
    # write_tif(LRPAN,'F:\GF2_Registration','LRPAN.tif')
    return fusion_MS,PAN_normalize,MS

def write_tif(img,save_name):
    if np.ndim(img)==3:
        bands=img.shape[2]
    else:
        bands=1
    if np.ndim(img)==2:
        img=np.expand_dims(img,axis=0)
    height, width= img.shape[0], img.shape[1]
    driver=gdal.GetDriverByName('GTiff')
    if os.path.exists(save_name):
        os.remove(save_name)
    save_tif=driver.Create(save_name,width,height,bands,gdal.GDT_UInt16)
    for i in range(bands):
        save_tif.GetRasterBand(i+1).WriteArray(img[:,:,i])

def ORB_match(MS,PAN,features_max,MS_origin):
    orb=cv2.ORB_create(nfeatures=features_max)
    kp_MS,des_MS=orb.detectAndCompute(MS,None)
    kp_PAN,des_PAN=orb.detectAndCompute(PAN,None)
    FLANN_INDEX_LSH=6
    index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12,multi_probe_level=1)
    search_params=dict(check=100)
    matcher=cv2.FlannBasedMatcher(index_params,search_params)
    matchs=matcher.match(des_MS,des_PAN)
    minDis=9999.0
    for m in matchs:
        if m.distance<minDis:
            minDis=m.distance
    good_matchs=[m for m in matchs if m.distance<(minDis*5)]
    MS_cor=[kp_MS[m.queryIdx].pt for m in good_matchs]
    PAN_cor=[kp_PAN[m.trainIdx].pt for m in good_matchs]
    AffineMatrix,mask=cv2.findHomography(np.array(MS_cor),np.array(PAN_cor),cv2.RANSAC)
    im=cv2.warpPerspective(MS_origin.transpose(1,2,0),AffineMatrix,(MS.shape[1],MS.shape[0]))
    write_tif(im,'RegisterMS.tif')
    cv2.imwrite('RegisterMS.tif',im)
    cv2.imshow('',MS-im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # point_pair=list(zip(MS_cor,PAN_cor))
    # point_pair.sort(key=lambda p:(p[0][0],p[0][1]))
    grid_index_MS=[(p[0]//81,p[1]//81) for p in MS_cor]
    grid_index_PAN=[(p[0]//81,p[1]//81) for p in PAN_cor]
    point_grid=defaultdict(list)
    for idx,GI in enumerate(grid_index_MS):
        if grid_index_PAN[idx]==GI:
            point_grid[GI].append((MS_cor[idx],PAN_cor[idx]))
    for grid,points in point_grid.items():
        MS_point = np.array([(p[0][0]-grid[0]*81,p[0][1]-grid[1]*81) for p in points]).astype(np.float32)
        PAN_point = np.array([(p[1][0]-grid[0]*81,p[1][1]-grid[1]*81) for p in points]).astype(np.float32)
        if len(points)>3:
            AffineMatrix=cv2.findHomography(MS_point,PAN_point,cv2.RANSAC)
            # AffineMatrix = cv2.getAffineTransform(MS_point,PAN_point)
            # x=(MS_origin[:, int(grid[0]) * 81:int(grid[0]) * 81 + 81, int(grid[1]) * 81:int(grid[1]) * 81 + 81]).transpose(2, 0, 1)
            AffineGrid=MS[int(grid[0]) * 81:int(grid[0]) * 81 + 81, int(grid[1]) * 81:int(grid[1]) * 81 + 81]#.transpose(1, 2, 0)
            MS_transfromed=cv2.warpPerspective(AffineGrid,AffineMatrix,(81,81))
            cv2.imshow('', np.concatenate((AffineGrid,MS_transfromed),axis=1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            MS_origin[:, int(grid[0]) * 81:int(grid[0]) * 81 + 81, int(grid[1]) * 81:int(grid[1]) * 81 + 81] = MS_transfromed.transpose(2,0,1)
        # elif len(points)>3:
        #     cv2.

    # grid_x,grid_y=np.meshgrid(range(0,MS.shape[0],3),range(0,MS.shape[1],3))
    # img_match=None
    # img_match=cv2.drawMatches(MS,kp_MS,PAN,kp_PAN,good_matchs,img_match)
    # cv2.imshow('',img_match)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__=='__main__':
    # test_img_path=r'F:\GF2_Registration\GF2_PMS1_E114.0_N22.6_20150108_L1A0000674009-MSS1_fusion_6_10.tif'
    MS_path=r'MS.tif'
    PAN_path=r'PAN.tif'
    MS,PAN,MS_origin=read_tif(MS_path,PAN_path)

    ORB_match(MS,PAN,5000,MS_origin)
    # cv2.imshow('',kp_MS)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # imgshpae=MS.shape
    # sift=sift_pyocl.SiftPlan(shape=imgshpae,dtype='uint16')
    # kp1=sift.keypoints(PAN)
    # kp2=sift.keypoints(MS)
    # kp1.sort(order=["scale", "angle", "x", "y"])
    # kp2.sort(order=["scale", "angle", "x", "y"])
    #
    # mp=sift_pyocl.MatchPlan(devicetype='CPU')
    # result=mp.match(kp1,kp2)
    # GCP_PAN=list(zip(result['x'][:,0],result['y'][:,0]))
    # GCP_MS=list(zip(result['x'][:,1],result['y'][:,1]))
    #
    #
    # alignPlan=sift_pyocl.LinearAlign(PAN)
    # MS_aligned=alignPlan.align(MS)
    # write_tif(MS_aligned,'F:\GF2_Registration','MS_aligned.tif')
    # write_tif(MS,'F:\GF2_Registration','MS_fusion.tif')

    # U, V = HornSchunck(PAN, MS_aligned)
    # write_tif(MS_aligned,'F:\GF2_Registration','MS_aligned.tif')
    # write_tif(MS,'F:\GF2_Registration','MS_fusion.tif')






