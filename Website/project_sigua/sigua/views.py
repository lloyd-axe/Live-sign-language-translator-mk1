from aiohttp import request
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse
from Modules.WebASLT import WebASLTranslator
from .models import Sentence

#############################################
MODEL_PATH = 'static/model.h5'
WORDLIST_PATH = 'static/word_list.txt'
pose = False
face = False
rightHand = False
leftHand = False
thres = 0.8
interval = 18
#############################################

sentenceVariable = Sentence
#generate feed
def gen(request, aslt):
    sentence, sequences = [], []
    frame_no = 0
    while True:
        frame, sentence, sequences = aslt.Stream(
            frame_no, 
            sentence, 
            sequences, 
            interval = interval, 
            threshold = thres, 
            draw = [pose, face, leftHand, rightHand])
        frame_no += 1
        if frame_no == interval + 1:
            frame_no = 0
        sentenceVariable.text = ' '.join(s for s in sentence)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def update(request):
    return JsonResponse({'sentence': str(sentenceVariable.text)})

def Home(request):
    #return HttpResponse(loader.get_template('sigua.html').render(RequestContext(request,{'test':t.test})))
    return render(request, 'sigua.html')

@gzip.gzip_page
def StreamVideo(request):
    with open(WORDLIST_PATH) as file:
        words = [line.rstrip() for line in file]
    
    aslt = WebASLTranslator(MODEL_PATH, words)
    #return render(request, 'sigua.html', {'test': "This is a test", 'stream': gen(aslt)})
    return StreamingHttpResponse(
        gen(request, aslt), 
        content_type="multipart/x-mixed-replace;boundary=frame")


