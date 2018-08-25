from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class TableSet(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	data = models.TextField()
	checkpoint = models.IntegerField(default=0)

class Credits(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	credits = models.IntegerField(default=10000)