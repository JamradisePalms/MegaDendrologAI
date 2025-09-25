from django.db import models


class Result(models.Model):
    event_id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500)
    date = models.DateTimeField(auto_now_add=True)
    tree_type = models.CharField(max_length=500)
    has_cracks = models.IntegerField()
    has_hollows = models.IntegerField()
    has_fruits_or_flowers = models.IntegerField()
    injuries = models.CharField(max_length=500)
    photo_file_name = models.CharField(max_length=500)
    answer = models.CharField(max_length=10000)


class Photo(models.Model):
    user_id = models.CharField(max_length=500)
    image = models.ImageField()
    uploaded_at = models.DateTimeField(auto_now_add=True)