from django.db import models

class Post (models.Model):
    title = models.CharField(max_length=50)
    description = models.TextField()

    def __str__(self):
        return self.title

class Calculation(models.Model):
    P = models.IntegerField()
    R = models.IntegerField()
    b1 = models.FloatField(default=0.1)
    b2 = models.FloatField(default=0.1)
    Sp = models.IntegerField(default=30)
    N = models.IntegerField(default=10)
    scb_alg_min = models.CharField(max_length=8)

