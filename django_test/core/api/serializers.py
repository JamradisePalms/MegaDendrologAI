from rest_framework import serializers
from api.models import Result, Photo


class EventInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result 
        fields = ("id", "probability", "species", "trunkRot",
                "trunkHoles", "trunkCracks", "trunkDamage",
                "crownDamage", "fruitingBodies", "diseases", "dryBranchPercentage",
                "additionalInfo", "overallCondition", "imageUrl", "imagePath",
                "analyzedAt", "isVerified")

class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = ("id", "user_id", "image", "uploaded_at")
        