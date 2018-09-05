from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class TableSet(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	data = models.TextField()
	checkpoint = models.IntegerField(default=0)
	free_service = models.IntegerField(default=0)
	test_store = models.TextField(default=0)
	test_cal = models.IntegerField(default=0)
	

class Credits(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	credits = models.IntegerField(default=75000)

class Logs(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	log = models.TextField()
	created_at = models.DateTimeField(auto_now_add=True)