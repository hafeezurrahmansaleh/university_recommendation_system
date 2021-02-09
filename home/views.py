from django.shortcuts import render, HttpResponse

from .predictor import Predictor
from .recommender import Recommender
from .models import *
from django.db.models import Q
from django.apps import apps

# Create your views here.

def home(request):
    return render(request, 'home/index.html')


def get_recommendation(request):
    return render(request, 'home/recommendation.html')


def university_list(request):
    if request.method == 'POST':
        # st_name = request.POST['name']
        # st_email = request.POST['email']
        major = request.POST['major']
        CGPA = float(request.POST['CGPA'])
        max_cgpa = float(request.POST['max_cgpa'])
        GREv = float(request.POST['GREv'])
        GREq = float(request.POST['GREq'])
        GREa = float(request.POST['GREa'])
        TOEFL = float(request.POST['TOEFL'])
        uni_rating = float(request.POST['uni_rating'])
        SOP = float(request.POST['SOP'])
        LOR = float(request.POST['LOR'])
        research = float(request.POST['research'])

        normalized_cgpa = CGPA / max_cgpa
        rec = Recommender()
        uni_list = rec.make_recommendation(GREv, GREq, GREa, normalized_cgpa)
        universities = University.objects.all()

        GRE_total = GREv + GREq + GREa
        if max_cgpa != 10:
            CGPA = (CGPA/max_cgpa)*10
        predictor_input = [[GRE_total, TOEFL, uni_rating, SOP, LOR, CGPA, research]]
        predictor = Predictor()
        chance_of_admit = predictor.get_prediction(predictor_input)

        context = {
            'universities': universities,
            'chance'      : chance_of_admit*100,
            'uni_list'      : uni_list,
        }
        return render(request, 'home/university-list.html', context)

    universities = University.objects.all()
    context = {
        'universities': universities
    }
    return render(request, 'home/university-list.html', context)


def university_details(request, uname):
    uni = University.objects.get(u_name=uname)
    context = {
        'uni_details': uni
    }
    return render(request, 'home/university-details.html', context)


def get_universities(u_names):
    # options = ['X1', 'X2', 'X3']
    qs = [Q(u_name__contains=option) for option in u_names]  # make a query for getting all the questions for every skill

    query = qs.pop()  # get the first element

    for q in qs:
        query |= q
    qs = University.objects.filter(query)
    return qs

