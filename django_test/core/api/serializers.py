from rest_framework import serializers
from api.models import Result, Photo


class EventInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result 
        fields = ("id", "probability", "species", "trunkRot",
                "trunkHoles", "trunkCracks", "trunkDamage",  "user_id",
                "crownDamage", "fruitingBodies", "overallCondition",
                "imageUrl", "imagePath", "analyzedAt", "isVerified", "plantName")

class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = ("id", "user_id", "image", "uploaded_at")
        
