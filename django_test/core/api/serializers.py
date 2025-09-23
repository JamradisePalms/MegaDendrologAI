from rest_framework import serializers
from api.models import Result


class EventInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result 
        fields = ("event_id", "date",
                "tree_type", "has_cracks",
                "has_hollows", "has_fruits_or_flowers",
                "injuries", "photo_file_name",
                "answer")
        