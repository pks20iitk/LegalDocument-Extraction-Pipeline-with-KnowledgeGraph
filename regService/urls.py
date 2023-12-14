from django.urls import path
from . import views
from django.conf import settings
 
urlpatterns = [
    path('api/urlragservice', views.urlresponse),
    path('api/getQuestions',views.getQuestions),
    path('api/getSuggestedQuestions',views.getSuggestedQuestion),
    path('api/generateKnowledge', views.knowledgeGraph),
    path('api/stream-url-rag-service', views.sse_view, name='stream-url-rag-service'),
]