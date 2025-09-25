from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views


urlpatterns = [
    path("add/<str:user_id>", views.add_raw, name="add_field"),
    path("show/<str:user_id>", views.get_table, name="show_table"),
    path("filter/<str:user_id>/<str:filters>/<int:count>", views.complex_filter, name="complex_filter"),
    path("sendphoto/<str:user_id>", views.save_file, name="send-photo"),
    path("photo/<str:user_id>", views.photo_list, name="photo"),
    path("photo/<str:user_id>/<int:pk>", views.photo_detail, name="photo")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

