from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

from home import views

urlpatterns = [
                  path('', views.home, name='home'),
                  path('get-recommendation/', views.get_recommendation, name='get_recommendation'),
                  path('university-list/', views.university_list, name='university_list'),
                  path('university-details/<uname>', views.university_details, name='university_details'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
