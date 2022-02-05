from django.db import models

# Create your models here.
class Sentence(models.Model):
    text = models.CharField(max_length=30)
