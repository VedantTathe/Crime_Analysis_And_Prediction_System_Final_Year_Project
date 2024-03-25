from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'main'

urlpatterns = [
    path('', views.index, name='index'),
    path('RegisterCrime.html/', views.registerCrime, name='registercrime'),
    path('addcrime/', views.addcrime, name='addcrime'),
    path('index.html/', views.index, name='index'),
    path('SearchCrime.html/', views.searchcrime, name='searchcrime'),
    path('UpdateCrime.html/', views.updatecrime, name='updatecrime'),
    path('searchcrime/', views.searchc, name='searchc'),
    path('ShowCrime.html/', views.show, name="show"),
    path('register.html/', views.register, name="registeruser"),
    path('login.html/', views.login, name="loginu"),
    path('updateuser.html/', views.updateuser, name="updateuser"),
    path('viewuser.html/', views.viewusers, name="viewusers"),
    path('chartjs.html/', views.chartjs, name="chartjs"),
    path('ViewCharts.html/', views.ViewCharts, name="ViewCharts"),
    # path('basic-table.html/', views.basictable, name="basictable"),
    path('viewcriminal.html/', views.viewcriminals, name="viewcriminal"),
    path('criminaldetail.html/', views.criminaldetail, name="criminaldetail"),
    path('showupdate_combined/', views.showupdate_combined, name='showupdate_combined'),
    path('UpdateCrime.html/<str:parameter>', views.updatecrime, name='updatecrime'),
    #path('showupdatecrime.html/', views.showcriminaldata, name='showcriminaldata'),
    path('showupdate/<str:param1>/<str:param2>/<str:param3>/', views.showupdate, name='showupdate'),
    path('showcrimedata/', views.showcrimedata, name='showcrimedata'),
    path('deletecrimedata/', views.deletecrimedata, name='deletecrimedata'),
    path('displaycrime/<str:param1>/', views.displaycrime, name='displaycrime'),
    path('displaycrime/', views.displaycrime, name='displaycrime'),
    # path('predicthotspot/', views.predicthotspot, name='predicthotspot'),
    path('searchchart.html/', views.searchchart, name='searchchart'),
    path('searchchartresult.html/', views.searchchartresult, name='searchchartresult'),
    path('welcomeMsg.html/', views.welcomeMsg, name='welcomeMsg'),
    # path('predicthotspotcall/', views.predicthotspotcall, name='predicthotspotcall'),
    path('searchcrimeresult.html/', views.searchcrimeresult, name='searchcrimeresult'),
    path('readFromVoiceSeparate/<str:param>/', views.readFromVoiceSeparate, name='readFromVoiceSeparate'),
    path('statuschangedata/', views.statuschangedata, name='statuschangedata'),
    path('statuschangedata/displaycrime/<int:param1>/', views.displaycrime, name='displaycrime'),
    path('futurehotspots.html/', views.futurehotspots, name='futurehotspots'),    
    path('predicthotspots_of_date/', views.predicthotspots_of_date, name='predicthotspots_of_date'),
    path('sampleurl/', views.sampleurl, name='sampleurl'),
    path('get_police_station/', views.get_police_station, name='get_police_station'),
    path('samplepage.html/', views.get_police_station, name='get_police_station'),
    path('predicthotspot/', views.predictHotspotsLatest, name='predictHotspotsLatest'),
    
    
    
   # path('UpdateCrime.html/showupdatecrime.html/<str:param1>/<str:param2>/<str:param3>/', views.showupdate, name="showupdate"),
]