window.onload = function() {
        data=[];
        showList();
        query();
        //var query_btn = $("#search_btn");
//        query_btn.click(function(){
//            query();
//
//        });
    }



function showList(){
    var num=1;
    $("#resource_tab").bootstrapTable({
        striped: true,//开启条纹
        locale:'zh-CN',//中文支持
        pagination: true,//是否开启分页（*）
        pageNumber:1,//初始化加载第一页，默认第一页
        pageSize: 10,//每页的记录行数（*）
        pageList: [10, 25, 50, 100],//可供选择的每页的行数（*）
        sidePagination: "client", //分页方式：client客户端分页，server服务端分页（*）
        showRefresh:false,//刷新按钮
        search: false,
        data:data,
        columns: [
            {field: 'SerialNumber',title: '序号', width:'10%',align:'center',formatter: function (value, row, index) {return index+1;}},
            {field: 'name', title:'影像资源', width:'10%', align:'center'},
            {field: 'area', title:'影像区域', width:'10%', align:'center'},
            {field: 'satelite', title:'卫星', width:'10%', align:'center'},
            {field: 'desc', title:'影像描述', width:'20%', align:'center'},
            {field: 'capture_time', title:'拍摄时间', width:'10%', align:'center'},
            {field: 'upload_time', title:'上传时间', width:'10%', align:'center'},
            {field: 'tool',title: '操作', align: 'center',
                          formatter: function (value,row,index){
                              var element = "<a href='#' class='operate delete_href' id='delete"+row.id +"' data-id='"+row.id +"' onclick='changeStatus(\" "+row.id+" \")' onmouseover='del_mouseOver(\" "+row.id+" \")' onmouseout='del_mouseOut(\" "+row.id+" \")'>"
                              + "<img id='del_img"+row.id+"' class='nav-img' src='../static/img/delete.png'>"
                              + "</a>";
                              return element;
                          },
                  }

        ],

    })
};

function check_mouseOver(data) {
    var num = parseInt(data);
    $("#check_img"+num).attr("src","../static/img/check1.png");
}

function check_mouseOut(data) {
    var num = parseInt(data);
    $("#check_img"+num).attr("src","../static/img/check.png");
}

function download_mouseOver(data) {
    var num = parseInt(data);
    $("#download_img"+num).attr("src","../static/img/download1.png");
}

function download_mouseOut(data) {
    var num = parseInt(data);
    $("#download_img"+num).attr("src","../static/img/download.png");
}

function del_mouseOver(data) {
    var num = parseInt(data);
    $("#del_img"+num).attr("src","../static/img/delete1.png");
}

function del_mouseOut(data) {
    var num = parseInt(data);
    $("#del_img"+num).attr("src","../static/img/delete.png");
}
function downloadImg(id){
    var globeID=parseInt(id);
    var button=$("#download"+globeID);
    $.ajax({
              type: 'POST',
              url: '/downloadImage/',
              data: {ImageID:globeID},
              success:function(message){
                  alert(message);
                  //button.removeAttr("disabled");
                  //if(message=="发布成功！")
                       //button.text("取消发布");
                  //else
                       //button.text("发布");
              },
              error:function(error){
                  alert(error);
              }
        });
}
function changeStatus(data){

    var num=data;
    $.ajax({
        url:'/delete_map/',
        type:'post',
        data: {'map_id':num},
        success: function(){
            alert('删除成功！');
             $("#resource_tab").bootstrapTable('remove',{
                field: 'id',
                values: [parseInt(data)],
            })
        }
    });
}
function deleteImg(id){
    var globeID=parseInt(id);
    var button=$("#del_img"+globeID);
    $.ajax({
              type: 'POST',
              url: '/deleteImage/',
              data: {ImageID:globeID},
              success:function(message){
                  alert(message);
                  //button.removeAttr("disabled");
                  //if(message=="发布成功！")
                       //button.text("取消发布");
                  //else
                       //button.text("发布");
              },
              error:function(error){
                  alert(error);
              }
        });
}
function release(data){
    var id=parseInt(data);
    var button=$("#release"+id);
    var isPublish=button.text();
    if(isPublish=="发布"){
        button.text("发布中");
        button.attr("disabled",true);
        $.ajax({
              type: 'POST',
              url: '/uploadImage/',
              data: {ImageID:id},
              success:function(message){
                  alert(message);
                  button.removeAttr("disabled");
                  if(message=="发布成功！")
                       button.text("取消发布");
                  else
                       button.text("发布");
              },
              error:function(error){
                  alert(error);
              }
        });
    }
    else if(isPublish=="取消发布"){
        button.text("正在取消");
        button.attr("disabled",true);
        $.ajax({
            type:'post',
            url: '/cancelPublish/',
            data: {ImageID:id},
            success:function(message){
                alert(message);
                button.removeAttr("disabled");
                if(message=="发布已取消！"){
                    button.text("发布");
                } else {
                    button.text("取消发布");
                }
            },
                error:function(error){
                    alert(error);
                }
            });
    }
}
function query(){
    var maptype= $("#maptype").val()

    $.ajax({
            type:'post',
            url:'/map_all/',
            data: {
                'maptype':maptype,
            },
            success:function(result){
                var data1=[];
                //result_data=JSON.parse(result['maps'])
                result_data=result['maps'];

                for(var i in result_data){
                    data1.push(result_data[i]);
                    }
              $("#resource_tab").bootstrapTable('load',data1);

                },
            error:function(){
                alert('搜索失败')}
         });
    }
function delete_res (data) {};

function show_map (data) {
    var id=parseInt(data);
    $.ajax({
              type: 'GET',
              url: '/rm_show_map/',
              data: {id:id},
              success:function(result){
                  var isinvalid=result['error'];
                  if(isinvalid)
                       alert('请先下载图像');
                  else
                       parent.window.document.getElementById("resource_management_container").src="/rm_show_map/?id="+id;
              },
              error:function(error){
                  alert(error);
              }
        });
};