from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib.auth import logout
from django.contrib.auth.decorators import permission_required
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.contrib import auth
from myweb.models import Module
from myweb.models import Buser
from myweb.models import Bmap
from myweb.models import Mask
import json
from django.contrib.gis.geos import GEOSGeometry ,Polygon
from django.core import serializers
from django.http import FileResponse
from django.utils import timezone
from django.forms.models import model_to_dict
from django.core.serializers.json import DjangoJSONEncoder
from django.http import HttpResponse, JsonResponse,StreamingHttpResponse
from wsgiref.util import FileWrapper
import os
from django.views.decorators.csrf import csrf_exempt
import gdal
from myweb.ImagryProcess import Preprocess
from django.contrib.gis.db.models.functions import Area,Transform
import time
from geoserver.catalog import Catalog
User = get_user_model()
MAPBASEPATH='/media/zhou/Document/yaogan/TJ'
mask_url='http://localhost:8080/geoserver/rest/workspaces/Mask/datastores/Mask/featuretypes'
map_url="http://localhost:8080/geoserver/rest/"
import requests
# Create your views here.
def index_new(request):
    return render(request,
                  template_name='index.html')


@csrf_exempt
def map_all(request):
    maps=Bmap.objects.all()

    map_data=[]
    for map in maps:
        mask_area=[]
        areas=Mask.objects.filter(map=map.id).order_by('type_id').values('area')
        #areas=Mask.objects.annotate(area=Area(Transform('mask',4527))).filter(map=map.id).order_by('type_id').values('area')
        for area in areas:
             #mask_area.append(round(area['area'].standard/1000000,2))
           mask_area.append(area['area'])
        center=map.polygon.centroid
        map=model_to_dict(map)
        map["map_area"] = sum(mask_area)
        map['center'] = [center[0], center[1]]
        map['mask_area']=mask_area
        map.pop("polygon")
        map_data.append(map)
    return JsonResponse({'maps':map_data})


@csrf_exempt
def map_compare(request):
    id1=request.POST.get("id1",False)
    id2 = request.POST.get("id2", False)
    interesting_area=request.POST.get("interesting_area", False)
    interesting_area_list=[]
    origin=Polygon()
    for ir in interesting_area:
        origin=origin.union(GEOSGeometry(ir))
    mask1_building = Mask.objects.get(map=id1, type_id=1).mask.union(origin)
    mask1_farm = Mask.objects.get(map=id1, type_id=5).mask.union(origin)
    mask1_forest = Mask.objects.get(map=id1, type_id=7).mask.union(origin)
    mask1_shack = Mask.objects.get(map=id1, type_id=6).mask.union(origin)
    mask1_water = Mask.objects.get(map=id1, type_id=4).mask.union(origin)
    mask1_grass = Mask.objects.get(map=id1, type_id=3).mask.union(origin)
    mask1_road = Mask.objects.get(map=id1, type_id=2).mask.union(origin)
    mask2_building = Mask.objects.get(map=id2, type_id=1).mask.union(origin)
    mask2_rest=Mask.objects.get(map=id2, type_id=0).mask.union(origin)
    demolition_area=(((mask1_building.union(mask1_farm)).union(mask1_forest)).union(mask1_shack)).intersection(mask2_rest)
    ibuild_area=(((mask1_grass.union(mask1_forest)).union(mask1_road)).union(mask1_water)).intersection(mask2_building)

    demolition_area_geojson={'type': 'Feature', 'geometry': demolition_area.geojson,'properties':{'type':'change'}}
    ibuild_area_geojson = {'type': 'Feature', 'geometry': ibuild_area.geojson, 'properties': {'type': 'change'}}
    return JsonResponse({"demolition_area":demolition_area_geojson,"ibuild_area":ibuild_area_geojson})


def index(request):
    return render(request,
                  template_name='index.html')


def account_show(request):
    users_temp = User.objects.all()
    d_users = {}
    for i in range(len(users_temp)):
        d_users[i] = model_to_dict(users_temp[i])
        user_permissions = []
        for j in range(len(d_users[i]['user_permissions'])):
            tmp = d_users[i]['user_permissions'][j].name
            user_permissions.append(tmp)
        d_users[i]['user_permissions'] = user_permissions
    if d_users:
        return render(request, 'account_Inquiry.html',
                      {'d_users': json.dumps(d_users, cls=DjangoJSONEncoder)})
    else:
        return render(request, 'account_Inquiry.html', {'message': '查找结果为空！'})


def account_inquiry(request):
    message = request.POST.get('message', False)
    fuzzy_search(message)
    users_temp=fuzzy_search(message)
    users = {}
    for i in range(len(users_temp)):
        users[i] = model_to_dict(users_temp[i])
        user_permissions = []
        for j in range(len(users[i]['user_permissions'])):
            tmp = users[i]['user_permissions'][j].name
            user_permissions.append(tmp)
        users[i]['user_permissions'] = user_permissions
    if users:
        return render(request, 'account_Inquiry.html', {'d_users': json.dumps(users, cls=DjangoJSONEncoder)})
    else:
        return render(request, 'account_Inquiry.html', {'message': '查找结果为空！'})


def add_Account(request):
    return render(request,
                  template_name='add_Account.html')


def add_usr(request):
    username = request.POST.get("username", False)
    enterprise_name= request.POST.get("enterprise_name", False)
    contact_usr= request.POST.get("contact_usr", False)
    phone= request.POST.get("phone", False)
    user_type = request.POST.get("user_type", False)
    check_box = request.POST.getlist('check_box', False)
    #check_box = json.loads(check_box)
    permission_dict = {'1': "city_management", '2': "agriculture_management", '3': "forestry_management",
                       '4': "environment_management",'5': "road_management",'6': "settlement_observation"}
    user = User.objects.create_user(username=username, enterprise_name=enterprise_name, usr_type=user_type,
                                    contact_usr=contact_usr, phone=phone)
    user.save()
    for i in check_box:
        permission = Permission.objects.get(codename=permission_dict[i])
        user.user_permissions.add(permission)
    return render(request,'add_Account.html',{'message':'添加成功'})


def _permissions_query(request):
    message = request.POST.get('message',False)
    query_method = request.POST.get('query_method', False)
    users_temp = []
    if query_method == '1':
        users_temp = User.objects.filter(username=message)
    if query_method == '2':
        users_temp = User.objects.filter(department_name=message)
    if query_method == '3':
        users_temp = User.objects.filter(phone=message)
    if query_method == '4':
        users_temp = User.objects.filter(contact_usr=message)
    users={}
    for i in range(len(users_temp)):
       users[i]=model_to_dict(users_temp[i])
       user_permissions = []
       for j in range(len(users[i]['user_permissions'])):
           tmp = users[i]['user_permissions'][j].name
           user_permissions.append(tmp)
       users[i]['user_permissions'] = user_permissions
    if users:
        return render(request,'permissions_query.html',locals())
    else:
        return render(request,'permissions_query.html',{'message1':'查找结果为空！'})



def upload_map(request):
    return render(request,
                  template_name='upload_map.html')
def password_revise(request):
    return render(request,
                  template_name='password_revise.html')

def query_map(request):

    return render(request,template_name='query_map.html')


def _upload_map(request):

    # resp=save_polygon('fusionRPC.tif')
    # return render(request, 'upload_map.html', {'message': resp})
    # try:
        map_name=request.POST.get("map_name",False)
        if Bmap.objects.filter(name=map_name):
            return render(request, 'upload_map.html', {'message': "地图已存在!"})
        map_area = request.POST.get("map_area", False)
        satelite= request.POST.get("satelite", False)
        desc=request.POST.get("desc", False)
        # wholemap= request.FILES.get('wholemap')
        t=time.strftime('%Y-%m-%d', time.localtime(time.time()))
        map=Bmap.objects.create(name=map_name,area=map_area,satelite=satelite,desc=desc,upload_time=time.strftime('%Y-%m-%d',time.localtime(time.time())))
        map.save()

        # f1=open(os.path.join(MAPBASEPATH,wholemap.name), 'wb')
        # for chunk in wholemap.chunks():
        #     f1.write(chunk)
        # f1.close()
        # tar=tarfile.open(os.path.join(MAPBASEPATH,wholemap.name))
        # target=os.path.join(MAPBASEPATH,str(map.id))
        # if os.path.exists(target):
        #     return render(request, 'upload_map.html', {'message': '文件已存在！'})
        # else:
        #     os.mkdir(target)
        # if not os.path.join(MAPBASEPATH,str(max(idlist)+1)):
        #     os.mkdir(target)
        # for file in tar:
        #     tar.extract(file,target)
        # tar.close()
        # Bmap.objects.filter(id=map.id).update(sourcefolder=os.path.join(MAPBASEPATH,map_name))
        return render(request,'upload_map.html',{'message':Preprocess.preprogress(map.id)})
    # except Exception as err:
    #     Bmap.objects.filter(id=map.id).delete()
    #     return render(request,'upload_map.html',{'message':str(err)})
@csrf_exempt
def delete_map(request):
    map_id=request.POST.get('map_id',False)
    cat = Catalog(map_url, 'admin', 'geoserver')
    if cat.get_layer('Map:' + map_id):
        cat.delete(cat.get_layer('Map:' + map_id))
        cat.reload()
    for label_type in range(1, 8):
        if cat.get_layer('Mask:' + map_id + '_' + str(label_type)):
            cat.delete(cat.get_layer('Mask:' + map_id + '_' + str(label_type)))
            cat.reload()
    try:
        if cat.get_store(map_id):
            st=cat.get_store(map_id)
            cat.delete(st)
            cat.reload()
    except Exception:
        pass
    if Bmap.objects.filter(id=map_id):
        map_name=Bmap.objects.get(id=map_id).name
        dir_root=os.path.join(MAPBASEPATH,map_name)
        delete_files=(map_id+".jpg",'chaneltransform.tif','chaneltransform_rpc.txt', 'chaneltransformRPC.txt', 'chaneltransformRPC.tif.ovr',
                      'label.tif','label_rpc.txt','labelRPC.tif')
        for file in delete_files:
            if os.path.exists(file):
                os.remove(os.path.join(dir_root,file))
        Bmap.objects.filter(id=map_id).delete()
    return HttpResponse("success")


def deliver_map(request):
    maps_temp = Mask.objects.all().order_by('-create_time')
    d_maps = {}
    for i in range(len(maps_temp)):
        d_maps[i] = model_to_dict(maps_temp[i])
    if d_maps:
      return HttpResponse(json.dumps({'d_maps': json.dumps(d_maps, cls=DjangoJSONEncoder)}))
    else:
      return HttpResponse(json.dumps({'d_maps': ''}))


def deliver_area(request):
    masks = Bmap.objects.all()
    areas = {}
    for mask in masks:
      areas[mask.type_id]=mask.mask.area
    return JsonResponse(areas)



def _download_map(request):
    mapid=request.GET.get("id",False)
    if mapid:
        map = Bmap.objects.get(id=mapid)
        file=map.downloadfile
        name=os.path.basename(file)
        wrapper = FileWrapper(open(file, 'rb'))
        response = HttpResponse(wrapper)
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename='+name
        return response


#   def file_iterator(file_name, chunk_size=512):
#      with open(file_name,'rb') as f:
#        while True:
#          c = f.read(chunk_size)
#          if c:
#            yield c
#          else:
#            break
#
# def _download_map(request):
#     mapid = request.GET.get("id", False)
#     map = Bmap.objects.get(id=mapid)
#     filename=map.wholemap_path
#     pathname=map.wholemap_path.split('.')[0]
#     response = StreamingHttpResponse(file_iterator(MAPBASEPATH+pathname+'/'+filename))
#     response['Content-Type'] = 'application/octet-stream'
#     response['Content-Disposition'] = 'attachment;filename="{0}"'.format(the_file_name)
#     return response

def _delete_map(request):
    mapid=request.GET.get("id", False)
    if mapid:
        map=Bmap.objects.get(id=mapid)
        map.delete()
        return JsonResponse({'result':'success'})
    else:
        return JsonResponse({'result':'error'})

def _delete_module(request):
        moduleid = request.GET.get("id", False)
        if moduleid:
            module = Module.objects.get(id=moduleid)
            module.delete()
            return JsonResponse({'result': 'success'})
        else:
            return JsonResponse({'result': 'error'})

def add_module(request):
    return render(request,'add_module.html')

def _add_module(request):
    module_name = request.POST.get("module_name", False)
    image = request.POST.get("image", False)
    purpose = request.POST.get("purpose", False)
    modify_time=timezone.now()
    is_active = request.POST.get("is_active", False)
    module = Module.objects.create(module_name=module_name, image=image, purpose=purpose,modify_time=modify_time,
                             is_active=is_active)
    module.save()
    return render(request, 'add_module.html', {'message': "success"})

def query_module(request):

    modules=Module.objects.all()
    d_modules = {}
    for i in range(len(modules)):
        d_modules[i] = model_to_dict(modules[i])
    return render(request,'query_module.html',{'d_modules': json.dumps(d_modules, cls=DjangoJSONEncoder)})

def info_revise(request):
    username=request.POST.get("username", False)
    return render(request, 'info_revise.html', {'username': username})

def sinfo_revise(request):
    username = request.session['username']
    user = User.objects.get(username=username)
    user = model_to_dict(user)
    return render(request, 'sinfo_revise.html', {'user': json.dumps(user, cls=DjangoJSONEncoder)})

def login(request):
    return render(request, template_name='login.html')


def login_check(request):
    username = request.POST.get("username", False)
    password= request.POST.get("password", False)
    user = auth.authenticate(username=username, password=password)
    if user:
        request.session['username']=username
        auth.login(request, user)
        return render(request, 'index.html', {'message1': '登录成功'})

    else:
       return render(request, 'login.html', {'message1': '用户名或密码错误'})


def _info_revise(request):
        username = request.POST.get("username", False)
        enterprise_name = request.POST.get("enterprise_name", False)
        contact_usr = request.POST.get("contact_usr", False)
        phone = request.POST.get("phone", False)
        usr_type=request.POST.get("usr_type", False)
        user = User.objects.get(username=username)
        if user:
            user.enterprise_name= enterprise_name
            user.usr_type = usr_type
            user.phone = phone
            user.contact_usr =contact_usr
            user.save()
            return render(request, 'info_revise.html', {'message': '修改成功！'})

        else:
            return render(request, 'info_revise.html',{'message': '用户不存在！'})

def _account_show(request):
    users_temp = User.objects.all()
    d_users = {}
    for i in range(len(users_temp)):
        d_users[i] = model_to_dict(users_temp[i])
        user_permissions = []
        for j in range(len(d_users[i]['user_permissions'])):
            tmp = d_users[i]['user_permissions'][j].name
            user_permissions.append(tmp)
        d_users[i]['user_permissions'] = user_permissions
    if d_users:
        return render(request, 'authorityManagement.html',{'d_users': json.dumps(d_users, cls=DjangoJSONEncoder)})
    else:
        return render(request, 'authorityManagement.html', {'message': '查找结果为空！'})

def status_revise(request):
    #raw_dic=request.raw_post_data()
    #dic=json.loads(raw_dic,cls=DjangoJSONEncoder)
    is_active=request.POST.get('is_active', False)
    id=request.POST.get('id', False)
    user=User.objects.get(id=id)
    user.is_active=is_active
    user.save()
    return HttpResponse("success")

def permission_revise(request):
    userid = request.POST.get("id", False)
    check_box = request.POST.get('permission_value',False)
    check_box=json.loads(check_box)
    user = User.objects.get(id=userid)
    user.user_permissions.clear()
    permission_dict={'1': "city_management", '2': "agriculture_management", '3': "forestry_management",
                       '4': "environment_management",'5': "road_management",'6': "settlement_observation"}
    for i in check_box:
       permission = Permission.objects.get(codename=permission_dict[i])
       user.user_permissions.add( permission )
    user=model_to_dict(user)
    user_permissions = []
    for j in range(len(user['user_permissions'])):
        tmp = user['user_permissions'][j].name
        user_permissions.append(tmp)
    user['user_permissions'] = user_permissions
    return HttpResponse(json.dumps({"user":user},cls=DjangoJSONEncoder))

def password_reset(request):
    old_password = request.POST.get("old_password",False)
    new_password = request.POST.get("new_password1",False)
    if request.user.check_password(old_password):
        request.user.set_password(new_password)
        request.user.save()

        return render(request, 'password_revise.html', {'message': '修改成功！'})
    else:
        return render(request, 'password_revise.html',{'message': '用户名或密码错误!'})

def _sinfo_revise(request):
        username = request.POST.get("username", False)
        enterprise_name = request.POST.get("enterprise_name", False)
        phone = request.POST.get("phone", False)
        user = User.objects.get(username=username)
        if user:
            user.enterprise_name = enterprise_name
            user.phone = phone
            user.save()
            return render(request, 'sinfo_revise.html', {'message': '修改成功！'})
        else:
            return render(request, 'sinfo_revise.html', {'message': '用户不存在！'})

def _add_superuser(request):
    password = request.POST.get("password1", False)
    username = request.POST.get("username", False)
    enterprise_name = request.POST.get("enterprise_name", False)
    contact_usr = request.POST.get("contact_usr", False)
    phone = request.POST.get("phone", False)
    user = User.objects.create_user(username=username, enterprise_name=enterprise_name, is_superuser=True,
                                    contact_usr=contact_usr, phone=phone,password=password)
    user.save()
    return render(request, 'add_superuser.html', {'message': '添加成功'})

def add_superuser(request):
    return render(request,template_name="add_superuser.html")

def mylogout(request):
    logout(request)
    return render(request,'login.html',{'islogin': False})

def check_username(request):
    username = request.POST.get('username', False)
    user=User.objects.filter(username=username)
    if user:
        return HttpResponse("false")
    else:
        return HttpResponse("true")
