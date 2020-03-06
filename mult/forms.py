from django import forms

class Multiply(forms.Form):
    P = forms.IntegerField(label='P = ', initial = 3)
    R = forms.IntegerField(label='R = ', initial = 2)
    b1 = forms.FloatField(label='b1 = ', initial = 0.1)
    b2 = forms.FloatField(label='b2 = ', initial = 0.1)
    Sp = forms.IntegerField(label='Sp = ', initial = 30)
    N = forms.IntegerField(label='N = ', initial = 10)