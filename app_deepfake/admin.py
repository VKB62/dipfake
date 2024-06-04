from django.contrib import admin
from .models import Deepfake


class DeepfakeAdmin(admin.ModelAdmin):
    list_display = ["id", "model", "result", "upload_at", "video"]
    list_filter = ["upload_at", "model", "result"]


admin.site.register(Deepfake, DeepfakeAdmin)
