from django.db import models
from smartfields import fields


# Create your models here.
class UniversityTag(models.Model):
    tag = models.CharField(null=True, max_length=50)


class University(models.Model):
    u_name = models.CharField(null=True, max_length=100)
    country = models.CharField(null=True, max_length=50)
    state = models.CharField(null=True, max_length=50)
    short_details = models.CharField(null=True, max_length=250)
    world_rank = models.IntegerField()
    tags = models.ManyToManyField(UniversityTag, null=True)
    image = fields.ImageField(upload_to='university')
