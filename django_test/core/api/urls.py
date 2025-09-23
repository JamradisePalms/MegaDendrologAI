from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views


urlpatterns = [
    path('add', views.add_raw, name="add_field"),
    path('show', views.get_table, name="show_table"),
    path('show/<str:field>/<str:column>', views.get_filtered_table, name="tree_type"),
    path('test/savefile', views.save_file, name="save_file"),
    path('filter/<str:filters>', views.complex_filter, name="complex_filter"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

