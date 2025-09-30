from django.db import models


class Result(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500)
    probability = models.FloatField()
    species = models.CharField(max_length=500)
    trunkRot = models.CharField(max_length=500)
    trunkHoles = models.CharField(max_length=500)
    trunkCracks = models.CharField(max_length=500)
    trunkDamage = models.CharField(max_length=500)
    crownDamage = models.CharField(max_length=500)
    fruitingBodies = models.CharField(max_length=500)
    diseases = models.CharField(max_length=500)
    dryBranchPercentage = models.FloatField()
    additionalInfo = models.CharField(max_length=500)
    overallCondition = models.CharField(max_length=500)
    imageUrl = models.CharField(max_length=500)
    imagePath = models.CharField(max_length=500)
    analyzedAt = models.DateTimeField(auto_now_add=True)
    isVerified = models.BooleanField()



    # event_id = models.AutoField(primary_key=True)
    # user_id = models.CharField(max_length=500)
    # date = models.DateTimeField(auto_now_add=True)
    # tree_type = models.CharField(max_length=500)
    # has_cracks = models.IntegerField()
    # has_hollows = models.IntegerField()
    # has_fruits_or_flowers = models.IntegerField()
    # injuries = models.CharField(max_length=500)
    # photo_file_name = models.CharField(max_length=500)
    # answer = models.CharField(max_length=10000)


class Photo(models.Model):
    user_id = models.CharField(max_length=500)
    image = models.ImageField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    url = models.CharField(max_length=500)