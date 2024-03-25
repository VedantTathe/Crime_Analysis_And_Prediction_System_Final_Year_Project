from django import forms
from django_recaptcha.fields import ReCaptchaField

class FormWithCaptcha(forms.Form):
    captcha = ReCaptchaField()
