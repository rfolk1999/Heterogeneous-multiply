# Generated by Django 2.2.10 on 2020-02-18 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mult', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('amount', models.DecimalField(decimal_places=4, max_digits=11)),
                ('datetime', models.DateTimeField()),
            ],
        ),
    ]