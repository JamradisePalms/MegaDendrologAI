from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse

from api.models import Result
from api.serializers import EventInfoSerializer

from django.core.files.storage import default_storage

from .module import num


@csrf_exempt
def get_filtered_table(request, field="", column=""):
    try:
        column = int(column)
    except:
        pass
    if request.method == 'GET':
        if field == "tree_type":
            event_info = Result.objects.filter(tree_type=column).order_by('event_id')[:10]
        elif field == "has_cracks":
            event_info = Result.objects.filter(has_cracks=column).order_by('event_id')[:10]
        elif field == "injuries":
            event_info = Result.objects.filter(injuries=column).order_by('event_id')[:10]
        elif field == "has_hollow":
            event_info = Result.objects.filter(has_hollow=column).order_by('event_id')[:10]
        elif field == "has_fruits_or_flowers":
            event_info = Result.objects.filter(has_fruits_or_flowers=column).order_by('event_id')[:10]
        else:
            return JsonResponse("No matching field", safe=False)
        event_info_serializer = EventInfoSerializer(event_info, many=True)
        return JsonResponse(event_info_serializer.data, safe=False)
    

@csrf_exempt
def get_table(request):
    if request.method == 'GET':
        event_info = Result.objects.all()
        event_info_serializer = EventInfoSerializer(event_info, many=True)
        return JsonResponse(event_info_serializer.data, safe=False)


@csrf_exempt
def add_raw(request):
    if request.method == 'POST':
        event_info_data = JSONParser().parse(request)
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


@csrf_exempt
def complex_filter(request, filters=""):
    if request.method == 'GET':
        event_info = Result.objects.all()
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
        #     if key == "tree_type":
        #         event_info = event_info.objects.filter(tree_type=value)
        #     elif filters[i] == "has_cracks":
        #         try:
        #             value = int(value)
        #         except:
        #             return JsonResponse("No matching field", safe=False)
        #         event_info = event_info.objects.filter(has_cracks=value)
        #     elif key == "injuries":
        #         event_info = event_info.objects.filter(injuries=value)
        #     elif key == "has_hollow":
        #         try:
        #             value = int(value)
        #         except:
        #             return JsonResponse("No matching field", safe=False)
        #         event_info = event_info.objects.filter(has_hollow=value)
        #     elif key == "has_fruits_or_flowers":
        #         try:
        #             value = int(value)
        #         except:
        #             return JsonResponse("No matching field", safe=False)
        #         event_info = event_info.objects.filter(has_fruits_or_flowers=filters[i + 1])
        #     else:
        #         return JsonResponse("No matching field", safe=False)
        event_info = event_info.order_by('event_id')[:10]
        event_info_serializer = EventInfoSerializer(event_info, many=True)
        return JsonResponse(event_info_serializer.data, safe=False)


@csrf_exempt
def save_file(request):
    event_info_data = num.get_data()
    event_info_serializer = EventInfoSerializer(data=event_info_data)
    if event_info_serializer.is_valid():
        event_info_serializer.save()
    
        file = request.FILES['file']
        file_name = default_storage.save(file.name, file)
        return JsonResponse(f"{file_name} added Successfully", safe=False)
