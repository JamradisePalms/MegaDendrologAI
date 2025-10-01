from django.db import models


class Result(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500)
    plantName = models.CharField(max_length=500)
    probability = models.FloatField()
    species = models.CharField(max_length=500)
    trunkRot = models.CharField(max_length=500)
    trunkHoles = models.CharField(max_length=500)
    trunkCracks = models.CharField(max_length=500)
    trunkDamage = models.CharField(max_length=500)
    crownDamage = models.CharField(max_length=500)
    fruitingBodies = models.CharField(max_length=500)
    overallCondition = models.CharField(max_length=500)
    imageUrl = models.CharField(max_length=500)
    imagePath = models.CharField(max_length=500)
    analyzedAt = models.DateTimeField(auto_now_add=True)
    isVerified = models.BooleanField()


class Photo(models.Model):
    user_id = models.CharField(max_length=500)
    image = models.ImageField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    url = models.CharField(max_length=500)