from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse

from api.models import Result, Photo
from api.serializers import EventInfoSerializer, PhotoSerializer

from django.http import FileResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

import datetime

from .module import num
    

@api_view(["GET"])
def get_table(request, user_id=""):
    '''
    Выдаёт все элементы бд с деревьями
    '''
    if request.method == 'GET':
        event_info = Result.objects.filter(user_id=user_id).order_by('event_id')
        event_info_serializer = EventInfoSerializer(event_info, many=True)
        return JsonResponse(event_info_serializer.data, safe=False)


@api_view(["POST"])
def add_raw(request, user_id=""):
    '''
    Буквально ручками добавляет поле в бд, в основном нужно только для тестов
    '''
    if request.method == 'POST':
        event_info_data = JSONParser().parse(request)
        event_info_data["user_id"] = user_id
        event_info_serializer = EventInfoSerializer(data=event_info_data)
        if event_info_serializer.is_valid():
            event_info_serializer.save()
            return JsonResponse("Added Successfully", safe=False)
        return JsonResponse("Failed to Add", safe=False)


# @csrf_exempt
# def eventInfoApi(request, id=0):
#     if request.method == 'GET':
#         event_info = Info.objects.all()
#         event_info_serializer = EventInfoSerializer(event_info, many=True)
#         return JsonResponse(event_info_serializer.data, safe=False)
    
#     elif request.method == 'POST':
#         event_info_data = JSONParser().parse(request)
#         event_info_serializer = EventInfoSerializer(data=event_info_data)
#         if event_info_serializer.is_valid():
#             event_info_serializer.save()
#             return JsonResponse("Added Successfully", safe=False)
#         return JsonResponse("Failed to Add", safe=False)
    
#     elif request.method == 'PUT':
#         event_info_data = JSONParser().parse(request)
#         event_info = Info.objects.get(EventInfoUserId=event_info_data['EventInfoUserId'])
#         event_info_serializer = EventInfoSerializer(event_info, data=event_info_data)
#         if event_info_serializer.is_valid():
#             event_info_serializer.save()
#             return JsonResponse("Updated Successfully", safe=False)
#         return JsonResponse("Failed to Update")
    
#     elif request.method == 'DELETE':
#         event_info = Info.objects.get(EventInfoUserId=id)
#         event_info.delete()
#         return JsonResponse("Deleted Successfully", safe=False)

fields = ("event_id", "date",
          "tree_type", "has_cracks",
          "has_hollows", "has_fruits_or_flowers",
          "injuries", "photo_file_name",
          "answer")


@api_view(["GET"])
def complex_filter(request, user_id="", filters="", count=0):
    '''
    На основании фильтров в юрле делает запрос к бд и возвращает результат
    '''
    if request.method == 'GET':
        event_info = Result.objects.filter(user_id=user_id).order_by('event_id')
        filters = filters.split("&")
        for i in range(len(filters)):
            filters[i] = filters[i].split("=")
            key = filters[i][0]
            if key not in fields:
                return JsonResponse("No matching fields", safe=True)
            value = filters[i][1]
            if "has_" in key:
                try:
                    value = int(value)
                except:
                    return JsonResponse("No matching field", safe=False)
            event_info = event_info.filter(**{key: value})
        event_info = event_info.order_by('event_id')
        if count < len(event_info):
            event_info = event_info[:count]
        event_info_serializer = EventInfoSerializer(event_info, many=True)
        return JsonResponse(event_info_serializer.data, safe=False)


@api_view(['POST'])
def save_file(request, user_id=""):
    '''
    Принимает фото, загружает его в бд, имитирует работу модели
    '''
    if request.method == "POST":
        event_info_data = num.get_data()
        event_info_data["user_id"] = user_id
        event_info_serializer = EventInfoSerializer(data=event_info_data)
        if event_info_serializer.is_valid():
            event_info_serializer.save()

            data = {"id": 0, "user_id": user_id, "image": request.data.copy()['file'], "uploaded_at": datetime.datetime.now()}
            print(data)
            serializer = PhotoSerializer(data=data)
            if serializer.is_valid():
                serializer.save()
                return JsonResponse("Success", safe=False)
        

@api_view(["GET"])
def photo_list(request, user_id):
    '''
    Отправляет список фоток из бд, пока нужно только для тестов
    '''
    photos = Photo.objects.filter(user_id=user_id).order_by("-uploaded_at")
    serializer = PhotoSerializer(photos, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def photo_detail(request, user_id, pk):
    '''
    Отправляет фотку, пока нужно только для тестов
    '''
    photo = get_object_or_404(Photo, pk=pk, user_id=user_id)
    return FileResponse(photo.image.open("rb"), content_type="image/jpeg")
