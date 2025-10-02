from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse

from api.models import Result, Photo
from api.serializers import EventInfoSerializer, PhotoSerializer

from django.http import FileResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.core.files import File
from django.conf import settings

import datetime
import locale
import os

from .module import main_onnx
    

@api_view(["GET"])
def get_table(request, user_id=""):
    '''
    Выдаёт все элементы бд с деревьями
    Args:
        request (HttpRequest): HTTP-запрос Django.
        user_id (str): Идентификатор пользователя.

    Returns:
        JsonResponse: JSON-массив записей в бд.
    '''
    if request.method == 'GET':
       event_info = Result.objects.filter(user_id=user_id).order_by('id')
       event_info_serializer = EventInfoSerializer(event_info, many=True)
       return JsonResponse(event_info_serializer.data, safe=False)


@api_view(["POST"])
def add_raw(request, user_id=""):
    '''
    Буквально ручками добавляет поле в бд, в основном нужно только для тестов
    Args:
        request (HttpRequest): HTTP-запрос Django с методом POST и JSON-данными.
        user_id (str): Идентификатор пользователя.

    Returns:
        JsonResponse:
            - Added Successfully - если сериализатор прошёл валидацию и запись сохранена.
            - Failed to Add - если сериализатор невалиден.
    '''
    if request.method == 'POST':
        event_info_data = JSONParser().parse(request)
        event_info_data["user_id"] = user_id
        event_info_serializer = EventInfoSerializer(data=event_info_data)
        if event_info_serializer.is_valid():
            event_info_serializer.save()
            return JsonResponse("Added Successfully", safe=False)
        return JsonResponse("Failed to Add", safe=False)
    

@api_view(["PUT"])
def edit(request, user_id=""):
    '''
    Обновление полей бд
    Args:
        request (HttpRequest): HTTP-запрос Django с методом PUT и JSON-данными.
                               Обязательное поле в теле запроса - id.
        user_id (str): Идентификатор пользователя.

    Returns:
        JsonResponse:
            - "Updated Successfully - если обновление прошло успешно.
            - Failed to Update - если сериализатор не прошёл валидацию.
            - Invalid user_id - если запись принадлежит другому пользователю.
    '''
    if request.method == 'PUT':
        request_info = JSONParser().parse(request)
        event_info = Result.objects.get(id=request_info["id"])
        if event_info.user_id == user_id:
            request_info["user_id"] = user_id
            event_info_serializer = EventInfoSerializer(event_info, data=request_info)
            if event_info_serializer.is_valid():
                event_info_serializer.save()
                return JsonResponse("Updated Successfully", safe=False)
            return JsonResponse("Failed to Update", safe=False)
        else:
            return JsonResponse("Invalid user_id", safe=False)
        



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

# fields = ("id", "date",
#           "tree_type", "has_cracks",
#           "has_hollows", "has_fruits_or_flowers",
#           "injuries", "photo_file_name",
#           "answer")

fields = ("id", "probability", "species", "trunkRot",
          "trunkHoles", "trunkCracks", "trunkDamage",
          "crownDamage", "fruitingBodies", "diseases", "dryBranchPercentage",
          "additionalInfo", "overallCondition", "imageUrl", "imagePath",
          "analyzedAt", "isVerified")

numeric_fields = ("probability", "dryBranchPercentage")

boolean_fields = ("isVerified")


@api_view(["GET"])
def complex_filter(request, user_id="", filters="", page=0):
    '''
    На основании фильтров в юрле делает запрос к бд и возвращает результат
    Args:
        request (HttpRequest): HTTP-запрос Django.
        user_id (str): Идентификатор пользователя.
        filters (str): Строка с фильтрами вида "field1=value1&field2=value2".
        count (int): Максимальное количество возвращаемых записей.
    Returns:
        JsonResponse: JSON-массив записей в бд.
    '''
    if request.method == 'GET':
        event_info = Result.objects.filter(id__range=((page - 1) * 10 , page * 10)).order_by('id')
        filters = filters.split("&")
        if filters == [" "] or filters == ["%20"]:
            event_info_serializer = EventInfoSerializer(event_info, many=True)
            return JsonResponse(event_info_serializer.data, safe=False)

        for i in range(len(filters)):
            filters[i] = filters[i].split("=")
            key = filters[i][0]
            if key not in fields:
                return JsonResponse("No matching fields", safe=True)
            value = filters[i][1]
            if key in numeric_fields:
                try:
                    value = float(value)
                except:
                    return JsonResponse("ValueError", safe=True)
            elif key in boolean_fields:
                try:
                    value = bool(value)
                except:
                    return JsonResponse("ValueError", safe=True)
            event_info = event_info.filter(**{key: value})
        event_info_serializer = EventInfoSerializer(event_info, many=True)
        return JsonResponse(event_info_serializer.data, safe=False)


@api_view(['POST'])
def save_file(request, user_id=""):
    '''
    Принимает фото, загружает его в бд
    Args:
        request (HttpRequest): HTTP-запрос Django.
        user_id (str): Идентификатор пользователя.

    Returns:
        JsonResponse: "Success" при успешной валидации и сохранении записи.
    '''
    if request.method == "POST":
            data = {"id": 0, "user_id": user_id, "image": request.FILES['file'],
                    "uploaded_at": datetime.datetime.now(), "url": request.build_absolute_uri(settings.MEDIA_URL + str(request.FILES['file']))}
            photo_serializer = PhotoSerializer(data=data)
            if photo_serializer.is_valid():
                photo_serializer.save()
            image_path = os.path.join(settings.MEDIA_ROOT, str(data["image"]))
            yolo = os.path.abspath(os.path.join("api", "module", "best.onnx"))
            tree_classifier = os.path.abspath(os.path.join("api", "module", "tree_type.onnx"))
            bad_things_classifier = os.path.abspath(os.path.join("api", "module", "everything_1.onnx"))
            cropped_image_path = settings.MEDIA_ROOT
            tree_type_result = main_onnx.run(image=image_path, yolo=yolo, classifier=tree_classifier, cropped_image_path=cropped_image_path)
            bad_things_result = main_onnx.run(image=image_path, yolo=yolo, classifier=bad_things_classifier, cropped_image_path=cropped_image_path)
            result = []

            for i in range(len(tree_type_result)):
                bad_things_result[i]["plantName"] = f"{tree_type_result[i]}, {bad_things_result[i]['plantName']}"
                bad_things_result[i]["species"] = tree_type_result[i]
                result.append(bad_things_result[i].copy())

            for i, el in enumerate(result):
                with open(os.path.join("photos", el["imageUrl"]), "rb") as f:
#                    url = request.build_absolute_uri(settings.MEDIA_URL + el["imageUrl"])
                    url = f"http://89.169.189.195:8080{settings.MEDIA_URL + el['imageUrl']}"
                    el["imageUrl"] = url
                    data = {"id": 0, "user_id": user_id, "image": File(f),
                        "uploaded_at": datetime.datetime.now(), "url": url}
                    photo_serializer = PhotoSerializer(data=data)
                    if photo_serializer.is_valid():
                        photo_serializer.save() # сохраненеи вырезанного дерева
                event_info_data = el.copy()
                event_info_data["user_id"] = user_id
                event_info_serializer = EventInfoSerializer(data=event_info_data)
                if event_info_serializer.is_valid():
                    event_info_serializer.save()
                else:
                    print(event_info_serializer.errors)
                result[i]["id"] = Result.objects.get(imageUrl=url).id
            
            return JsonResponse(result, safe=False)
        

@api_view(["GET"])
def photo_list(request, user_id):
    '''
    Отправляет список фоток из бд, пока нужно только для тестов
    Args:
        request (HttpRequest): HTTP-запрос Django.
        user_id (str): Идентификатор пользователя.

    Returns:
        Response: JSON-список фотографий.
    '''
    photos = Photo.objects.filter(user_id=user_id).order_by("-uploaded_at")
    serializer = PhotoSerializer(photos, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def photo_detail(request, user_id, pk):
    '''
    Отправляет фотку, пока нужно только для тестов
    Args:
        request (HttpRequest): HTTP-запрос Django.
        user_id (str): Идентификатор пользователя.
        pk (int): Первичный ключ записи 'Photo'.
    Returns:
        JsonResponse: Строка с абсолютным URL фотографии.
    '''
    photo = get_object_or_404(Photo, pk=pk, user_id=user_id)

    url = request.build_absolute_uri(settings.MEDIA_URL + str(photo.image))
    return JsonResponse(url, safe=False)
