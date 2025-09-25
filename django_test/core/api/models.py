from django.db import models

# Create your models here.
class Result(models.Model):
    event_id = models.AutoField(primary_key=True)
    date = models.DateTimeField(auto_now_add=True)
    tree_type = models.CharField(max_length=500)
    has_cracks = models.IntegerField()
    has_hollows = models.IntegerField()
    has_fruits_or_flowers = models.IntegerField()
    injuries = models.CharField(max_length=500)
    photo_file_name = models.CharField(max_length=500)
    answer = models.CharField(max_length=10000)
    