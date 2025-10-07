

#dir_base = 'C:\\Users\\wilbert\\Documents\\Wilbert_git\\Ai-FindMyPhoto'

'''

#### 1. (re)load the code on python prompt
####
#### Note: modify 'dir_base' and run these lines your python prompt

dir_base = 'C:\\Users\\wilbert\\Documents\\Wilbert_git\\Ai-FindMyPhoto'
exec(open( dir_base + '/FindMyPhoto.py' ).read())

#### 2. download example files --> ./input/*.jpg
####
#### Note: these are free-to-use photos of the dutch royal family

download_files(url_prefix, url_list, dir_input)

#### 3. collect person faces --> ./input_knownpersons/*.jpg
####
#### Let it find some faces you can use as 'known faces', stop it by pressing CTRL+C
#### Identify and move some of the faces --> ./input_knownpersons/<person>/*.jpg
#### 
#### Note: you don't need to do this as there are some included in the project

collect_person_faces()

#### 5. recognize faces --> ./output_found/*
####
#### results are shown until a next result is found, stop it by pressing CTRL+C

recognize_faces_show()

'''

import os
from pathlib import Path
import datetime
import shutil
import requests
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

import cv2
import torch
from deepface import DeepFace


# pprint ############################################

#formatted dictionary print
#https://docs.python.org/3/library/pprint.html
from pprint import pprint
#pprint({'score': np.float64(1.0), 'facial_area': [np.int64(0), np.int64(2265), np.int64(0), np.int64(2265)]})


def log(*args):
    s1=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(s1,args)
    with open(logfilename, 'a', encoding='utf-8') as f:
        pprint({'time':s1,'msg':args}, f, width=160, compact=True, sort_dicts=False)

debug=1
dir_input = os.path.join(dir_base, 'input')
dir_output = os.path.join(dir_base, 'output_found')
dir_known_persons = os.path.join(dir_base, 'input_knownpersons')
logfilename=os.path.join(dir_base,'FindMyPhoto.LOG.py')
face_minimum_size=40
face_minimum_size_collect=100
recognize_minimum_confidence=55

# os.environ['TF_USE_LEGACY_KERAS'] = '1'
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log('available device:',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def show_subplots(subplotsarray):
    rows=len(subplotsarray)
    cols=2
    for r in range(rows):
        cols=int(np.maximum(cols,len(subplotsarray[r])))
    rows=int(np.maximum(rows,2))
    #Note: subplots with cols=1 or rows=1 returns a 1-dimensional array, so >=2
    fig, axs = plt.subplots(ncols=cols, nrows=rows)

    for r in range(rows):
        for c in range(cols):
            axs[r,c].axis('off')
            if r<len(subplotsarray) and c<len(subplotsarray[r]):
                if 'title' in subplotsarray[r][c]:
                    axs[r,c].set_title(subplotsarray[r][c]['title'], pad=0)
                if 'img' in subplotsarray[r][c]:
                    #Note: OpenCV uses Blue-Green-Red order, convert to RGB for matplotlib
                    axs[r, c].imshow(cv2.cvtColor(subplotsarray[r][c]['img'], cv2.COLOR_BGR2RGB))
                if 'label1' in subplotsarray[r][c]:
                    axs[r,c].text(0.5, 0.8, subplotsarray[r][c]['label1'], horizontalalignment='center', verticalalignment='top', transform=axs[r,c].transAxes)
                if 'label2' in subplotsarray[r][c]:
                    axs[r,c].text(0.5, 0.5, subplotsarray[r][c]['label2'], horizontalalignment='center', verticalalignment='center', transform=axs[r,c].transAxes)
                if 'label3' in subplotsarray[r][c]:
                    axs[r,c].text(0.5, 0.2, subplotsarray[r][c]['label3'], horizontalalignment='center', verticalalignment='bottom', transform=axs[r,c].transAxes)
                if 'emphasize' in subplotsarray[r][c]:
                    if subplotsarray[r][c]['emphasize']>=1:
                        axs[r,c].axis('on')
                        axs[r,c].get_xaxis().set_visible(False)
                        axs[r,c].get_yaxis().set_visible(False)                        
                        if subplotsarray[r][c]['emphasize']>=2:
                            axs[r,c].set_facecolor('lightgreen')
                        else:
                            axs[r,c].set_facecolor('lightyellow')

    plt.get_current_fig_manager().resize(cols*150,rows*150)
    plt.show(block=False)
    plt.pause(0.001) #allow to window to update


#face is padded, to be used as reference
def padded_face(image, x, y, w, h):

    #padding
    padding=(w+h)//4

    imgh, imgw, imgc = image.shape

    #limit padding to make sure it doesnt go out of bounds
    if x-padding<0: padding=x
    if y-padding<0: padding=y
    if x+w+padding>imgw: padding=imgw-x-w
    if y+h+padding>imgh: padding=imgh-y-h

    img_padded = image[y-padding:y+h+padding, x-padding:x+w+padding]

    return img_padded


def deep_faces(file):
    padded_faces_dict = {}
    padded_faces_dict['file']=str(file)

    image=cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_UNCHANGED)
    imageh, imagew, imagec = image.shape
    padded_faces_dict['h']=imageh
    padded_faces_dict['w']=imagew
    padded_faces_dict['c']=imagec

    #detect faces
    errors=0
    faces = DeepFace.extract_faces( image
        , detector_backend='retinaface' # retinaface, mtcnn, ...
        , enforce_detection=False       # raise error if no face is found
        , align=True                    # align face based on eyes position
        , normalize_face=False          # 0..255 color values
        , color_face='bgr'              # blue-green-red color order
        )

    facesarray=[]
    for i,face in enumerate(faces):
        facial_area = face['facial_area']
        x,y,w,h=facial_area['x'],facial_area['y'],facial_area['w'],facial_area['h']
        left_eye = facial_area['left_eye']
        right_eye = facial_area['right_eye']
        obj={'x':x, 'y':y, 'w':w, 'h':h,
            'score' : face['confidence']*100}

        #empty, whole image or too small
        if h==0 or w==0:
            log('ERROR: face=', i, 'facial_area=', h, 'x', w, 'empty')
            errors+=1
            continue
        if h<face_minimum_size or w<face_minimum_size:
            log('ERROR: face=', i, 'facial_area=', h, 'x', w, 'too small')
            errors+=1
            continue
        if h+5>imageh or w+5>imagew:
            log('ERROR: face=', i, 'facial_area=', h, 'x', w, 'whole image')
            errors+=1
            continue

        log('found: face=', i, 'facial_area=', h, 'x', w, 'whole image')

        if 'mouth_right' in facial_area:
            obj['mouth_right']=facial_area['mouth_right']
        if 'mouth_left' in facial_area:
            obj['mouth_left']=facial_area['mouth_left']
        if 'left_eye' in facial_area:
            obj['left_eye']=facial_area['left_eye']
        if 'right_eye' in facial_area:
            obj['right_eye']=facial_area['right_eye']
        if 'nose' in facial_area:
            obj['nose']=facial_area['nose']

        obj['img_final'] = face['face']
        obj['img_padded'] = padded_face(image, x, y, w, h)
        facesarray.append(obj)

    log('image', imagew, 'x', imageh, errors, 'errors', len(faces)-errors, 'faces')

    padded_faces_dict['errors'] = errors
    padded_faces_dict['facesarray'] = facesarray

    return padded_faces_dict


def normalize_name(name):
    return name.replace('\\','-') \
        .replace('/','-') \
        .replace(':','-') \
        .replace('-','-') \
        .replace('_','-') \
        .replace('--','-')


def normalize_filename(dir_input,file):
    file=Path( file )
    #file dir
    n_file_dir='/'+normalize_name(str(file.parent)+'-')
    #root input dir
    n_input_dir='/'+normalize_name(dir_input+'-')
    #remove root input dir
    n_file_dir=n_file_dir.replace(n_input_dir,'').replace('/','')
    return normalize_name(n_file_dir+file.stem)


def mark_image(image,label,face,color):
    font_scale=1.7
    font = cv2.FONT_HERSHEY_PLAIN
    thick=1

    #detected_face = image[int(y):int(y+h), int(x):int(x+w)]
    x,y,w,h=face['x'],face['y'],face['w'],face['h']
    radius=(w+h)//40
    if debug>=1:
        log(label,'(',x,',',y,') ',w,'x',h)

    cv2.rectangle(image, (x, y), (x+w, y+h), color=color, thickness=thick)

    if 'mouth_left' in face and 'mouth_right' in face :
        pt1=(round(face['mouth_left'][0]), round(face['mouth_left'][1]))
        pt2=(round(face['mouth_right'][0]),round(face['mouth_right'][1]))
        cv2.line(image, pt1, pt2, color=color, thickness=thick)

    if 'nose' in face:
        pt3=(round(face['nose'][0]),round(face['nose'][1]))
        cv2.circle(image, pt3, radius, color=color, thickness=thick)

    if 'left_eye' in face:
        pt4=(round(face['left_eye'][0]),round(face['left_eye'][1]))
        cv2.circle(image, pt4, radius, color=color, thickness=thick)

    if 'right_eye' in face:
        pt5=(round(face['right_eye'][0]),round(face['right_eye'][1]))
        cv2.circle(image, pt5, radius, color=color, thickness=thick)

    (txtw, txth) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=thick)[0]
    cv2.rectangle(image, (x, y), (x + txtw + 2, y - txth - 2), color, cv2.FILLED)
    cv2.putText(image, label, (x, y), font, fontScale=font_scale, color=(0, 0, 0), thickness=thick)


def show_faces_marked(image,faces):
    # Mark the detected faces and show the image
    for i,face in enumerate(faces['facesarray']):
        if 'confidence' in face and face['confidence']>recognize_minimum_confidence:
            mark_image(image,'face' + str(i) + ' ' + str(round(face['confidence'])) + '%',face,(0, 255, 0))
        else:
            mark_image(image,'face' + str(i),face,(0, 255, 255))
        
    if faces['w']>faces['h']:
        resized = resize_with_aspect_ratio(image, width=1440)
    else:
        resized = resize_with_aspect_ratio(image, height=1080)
    cv2.imshow('Face Detection', resized)
    cv2.waitKey(1) # allow window to update

def close_all_windows():
    cv2.destroyAllWindows()
    plt.close('all')


def read_known_faces(dir_known_persons):
    #known faces
    # \dir_known_persons\<name>\*.jpg
    known_faces = {}
    nfaces=0
    for folder in list(Path(dir_known_persons).glob('*/')):
        person=normalize_name(str(folder)).split('-')[-1]
        known_faces[person]=[]
        for file in list(Path(folder).glob('*.jpg')):
            if file.is_file and not file.stem.endswith('-padded'):
                known_faces[person].append(
                    {   'name': file.stem,
                        'img': cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_UNCHANGED)
                    })
                nfaces+=1
    log('known',len(known_faces), 'persons', nfaces, 'faces')
    return known_faces


# Collect person faces #################################################################
#
# collect faces from a number of photos. dir_input -> dir_known_persons

def collect_person_faces():

    allimages = []
    for file in list(Path(dir_input).rglob('*.*')):
        if file.is_file and file.suffix.lower() in ['.jpg', '.jpeg']:
            allimages.append(file)

    n=len(allimages)
    log('allimages:',n)

    #random order, so it doesnt start with the same photos with each run
    nindexes=np.random.choice(range(len(allimages)), size=n, replace=False)
    nimages=[ allimages[i] for  i in nindexes ]

    #extract faces
    mostfacesfile=None
    mostfaces=-1
    for i, file in enumerate(nimages):
        nname=normalize_filename(dir_input,file)

        if list(Path(dir_output).glob(nname+'-*.jpg')):
            log(i,'/',n,'SKIPPED',file)
        else:
            log(i,'/',n,file)
            faces=deep_faces(file)
            for j,face in enumerate(faces['facesarray']):
                h,w,c=face['img_final'].shape
                if h>face_minimum_size_collect and w>face_minimum_size_collect:
                    pfilename=os.path.join(dir_known_persons, nname)
                    cv2.imwrite(pfilename+str(j)+'-padded.jpg', face['img_padded'])
                    cv2.imwrite(pfilename+str(j)+'.jpg', face['img_final'])
                    with open(pfilename+str(j)+'.py', 'w', encoding='utf-8') as f:
                        pprint({'numfaces' : len(faces['facesarray']),
                                'score':round(face['score']*100),
                                'file':str(file)}, f, width=160, compact=True, sort_dicts=False)


# Download example files #########################################################
#
# Free-to-use example files

'''
The copyright of these photos lies with the RVD. They may be downloaded 
free of charge for editorial use by news media, display in public spaces, 
private use, and educational purposes. 
Source: https://www.koninklijkhuis.nl/foto-en-video/fotos-koninklijk-gezin
'''

url_prefix='https://www.koninklijkhuis.nl/binaries/large/content/gallery/koninklijkhuis/content-afbeeldingen/'

url_list='''
nieuws/2016/02/lech-fotosessie-1-groot.jpg
nieuws/2016/02/lech-fotosessie-2-groot.jpg
nieuws/2016/02/lech-fotosessie-3-groot.jpg
nieuws/2019/12/2019-koninklijk-gezin-sevilla.jpeg
nieuws/2020/12/kerstfoto-2020-koninklijk-gezin.jpg
nieuws/2020/12/kerstkaart-koninklijk-gezin-2020.jpg
nieuws/2021/12/kerstkaart-2021.jpg
nieuws/2021/12/kerstkaart.jpeg
nieuws/2022/12/familie-kerst-2022.jpg
nieuws/2022/12/kerstkaart-2022.jpg
portretfoto-s/fotosessies/2007/de-prins-van-oranje-prinses-maxima-met-hun-dochters-2007---1.jpg
portretfoto-s/fotosessies/2007/de-prins-van-oranje-prinses-maxima-met-hun-dochters-2007---2.jpg
portretfoto-s/fotosessies/2008/de-prins-van-oranje-prinses-maxima-en-hun-dochters-2008---1.jpg
portretfoto-s/fotosessies/2008/de-prins-van-oranje-prinses-maxima-en-hun-dochters-2008---2.jpg
portretfoto-s/fotosessies/2010/de-prins-van-oranje-prinses-maxima-en-hun-kinderen-in-argentinie-2010.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---1.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---2.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---3.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---4.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---5.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---6.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---7.jpg
portretfoto-s/fotosessies/2010/fotosessie-landgoed-de-horsten-juli-2010---8.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---1.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---2.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---3.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---4.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---5.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---6.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---7.jpg
portretfoto-s/fotosessies/2012/fotosessie-van-de-prins-van-oranje-en-zijn-gezin-2012---8.jpg
portretfoto-s/fotosessies/2013/de-prins-van-oranje-met-gezin-naar-oostenrijk.jpg
portretfoto-s/fotosessies/2013/fotosessie-gezin-koning-willem-alexander-2013---1.jpg
portretfoto-s/fotosessies/2013/fotosessie-gezin-koning-willem-alexander-2013---2.jpg
portretfoto-s/fotosessies/2013/fotosessie-gezin-koning-willem-alexander-2013---3.jpg
portretfoto-s/fotosessies/2013/fotosessie-gezin-koning-willem-alexander-2013---4.jpg
portretfoto-s/fotosessies/2013/fotosessie-gezin-koning-willem-alexander-2013---5.jpg
portretfoto-s/fotosessies/2013/fotosessie-gezin-koning-willem-alexander-2013---6.jpg
portretfoto-s/fotosessies/2013/koning-willem-alexander-en-koningin-maxima-met-de-prinsessen-catharina-amalia-alexia-en-ariane.jpg
portretfoto-s/fotosessies/2013/prinses-alexia-prinses-catharina-amalia-en-prinses-ariane.jpg
portretfoto-s/fotosessies/2013/winterfotosessie-in-lech-2013---1
portretfoto-s/fotosessies/2013/winterfotosessie-in-lech-2013---2
portretfoto-s/fotosessies/2015/koning-koningin-dochters-en-honden-skipper-en-nala-fotosessie-2015.jpg
portretfoto-s/fotosessies/2015/koning-koningin-en-dochters-fotosessie-2015.jpg
portretfoto-s/fotosessies/2015/koning-willem-alexander-en-koningin-maxima-fotosessie-2015.jpg
portretfoto-s/fotosessies/2015/prinses-catharina-amalia-met-hond-nala-fotosessie-2015.jpg
portretfoto-s/fotosessies/2015/prinsessen-catharina-amalia-alexia-en-ariane-fotosessie-2015.jpg
portretfoto-s/fotosessies/2016/kerst/kerstfoto-2016-koninklijk-gezin.jpg
portretfoto-s/fotosessies/2016/originelen-zomerfotosessie/3-prinsessen-fotosessie-eikenhorst-2016-origineel.jpg
portretfoto-s/fotosessies/2016/originelen-zomerfotosessie/koning-en-koningin-fotosessie-eikenhorst-2016-origineel.jpg
portretfoto-s/fotosessies/2016/originelen-zomerfotosessie/koning-en-prinses-van-oranje-fotosessie-eikenhorst-2016-origineel.jpg
portretfoto-s/fotosessies/2016/originelen-zomerfotosessie/koninklijk-gezin-fotosessie-eikenhorst-2016-origineel.jpg
portretfoto-s/fotosessies/2017/kerstkaart/koninklijk-gezin-kerstfoto-2017.jpg
portretfoto-s/fotosessies/2017/lech/fotosessie-lech-2017-groepsfoto.jpg
portretfoto-s/fotosessies/2017/lech/fotosessie-lech-2017-koning-prinses.jpg
portretfoto-s/fotosessies/2017/lech/fotosessie-lech-2017-skiend.jpg
portretfoto-s/fotosessies/2017/zomer/originelen/drie-prinsessen-fotosessie-zomer-2017-origineel.jpg
portretfoto-s/fotosessies/2017/zomer/originelen/koning-en-koningin-fotosessie-zomer-2017-origineel.jpg
portretfoto-s/fotosessies/2017/zomer/originelen/koninklijk-gezin-fotosessie-zomer-2017-origineel.jpg
portretfoto-s/fotosessies/2017/zomer/originelen/koninklijk-gezin-in-sloep-fotosessie-zomer-2017-origineel.jpg
portretfoto-s/fotosessies/2017/zomer/originelen/koninklijk-gezin-met-molen-fotosessie-zomer-2017-origineel.jpg
portretfoto-s/fotosessies/2018/2018-erwin-olaf/gezinsfoto---erwin-olaf---2018px.jpg
portretfoto-s/fotosessies/2018/2018-erwin-olaf/koning-willem-alexander-koningin-maxima-prinses-van-oranje-prinses-alexia-en-prinses-ariane---erwin-olaf---2018---staand.jpg
portretfoto-s/fotosessies/2018/lech/fotosessie-lech-2018-highres-2.jpg
portretfoto-s/fotosessies/2018/lech/fotosessie-lech-2018-highres-3.jpg
portretfoto-s/fotosessies/2018/lech/fotosessie-lech-2018-highres-4.jpg
portretfoto-s/fotosessies/2018/zomer-2018/koning-willem-alexander-de-prinses-van-oranje-prinses-alexia-koningin-maxima-en-prinses-ariane-voor-de-eikenhorst-zomer-2018.jpg
portretfoto-s/fotosessies/2018/zomer-2018/koning-willem-alexander-de-prinses-van-oranje-prinses-alexia-koningin-maxima-en-prinses-ariane-zomer-2018.jpg
portretfoto-s/fotosessies/2018/zomer-2018/koning-willem-alexander-en-koningin-maxima-zomer-2018.jpg
portretfoto-s/fotosessies/2018/zomer-2018/koning-willem-alexander-koningin-maxima-prinses-catharina-amalia-prinses-alexia-en-prinses-ariane-zomer-2018.jpg
portretfoto-s/fotosessies/2018/zomer-2018/prinses-alexia-de-prinses-van-oranje-en-prinses-ariane-zomer-2018.jpg
portretfoto-s/fotosessies/2019/lech/fotosessie-lech-2019-koning-willem-alexander-en-koningin-maxima.jpg
portretfoto-s/fotosessies/2019/lech/fotosessie-lech-2019-koninklijk-gezin.jpg
portretfoto-s/fotosessies/2019/lech/fotosessie-lech-2019-koninklijke-familie.jpg
portretfoto-s/fotosessies/2019/zomer-2019/zomerfotosessie-2019-1-koninklijk-gezin.jpg
portretfoto-s/fotosessies/2019/zomer-2019/zomerfotosessie-2019-2-prinsessen.jpg
portretfoto-s/fotosessies/2019/zomer-2019/zomerfotosessie-2019-3-koninklijk-gezin.jpg
portretfoto-s/fotosessies/2019/zomer-2019/zomerfotosessie-2019-4-koninklijk-gezin-4000px.jpg
portretfoto-s/fotosessies/2019/zomer-2019/zomerfotosessie-2019-5-koninklijk-gezin-4000px.jpg
portretfoto-s/fotosessies/2019/zomer-2019/zomerfotosessie-2019-6-koninklijk-paar-4000px.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-koning-koningin-1.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-koning-koningin-2.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-koning-kroonprinses-prinses-beatrix.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-koninklijke-familie.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-prinses-beatrix-kleinkinderen.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-prinsessen-1.jpg
portretfoto-s/fotosessies/2020/lech/fotosessie-lech-2020-prinsessen-2.jpg
portretfoto-s/fotosessies/2020/zomer-2020/zomerfotosessie-gezin-koning-willem-alexander.jpg
portretfoto-s/fotosessies/2020/zomer-2020/zomerfotosessie-koning-willem-alexander-konigin-maxima.jpg
portretfoto-s/fotosessies/2020/zomer-2020/zomerfotosessie-prinses-alexia-prinses-van-oranje-prinses-ariane.jpg
portretfoto-s/fotosessies/2021/zomer-2021/fotosessie-zomer-2021-drie-prinsessen.jpg
portretfoto-s/fotosessies/2021/zomer-2021/fotosessie-zomer-2021-koning-willem-alexander-koningin-maxima.jpg
portretfoto-s/fotosessies/2021/zomer-2021/fotosessie-zomer-2021-koninklijk-gezin.jpg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/de-prinses-van-oranje-en-koning-willem-alexander.jpeg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/de-prinses-van-oranje-koningin-maxima-prinses-ariane-en-prinses-alexia-1.jpeg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/de-prinses-van-oranje-koningin-maxima-prinses-ariane-en-prinses-alexia-2.jpeg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/koning-willem-alexander-en-koningin-maxima.jpeg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/koning-willem-alexander-koningin-maxima-de-prinses-van-oranje-prinses-alexia-en-prinses-ariane.jpeg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/prinses-ariane-de-prinses-van-oranje-en-prinses-alexia-1.jpeg
portretfoto-s/fotosessies/2022/de-nieuwe-kerk-amsterdam-2022/prinses-ariane-de-prinses-van-oranje-en-prinses-alexia-2.jpeg
portretfoto-s/fotosessies/2022/zomer-2022/koning-willem-alexander-en-de-prinses-van-oranje-tijdens-de-fotosessie-op-paleis-noordeinde.jpg
portretfoto-s/fotosessies/2022/zomer-2022/koning-willem-alexander-en-koningin-maxima-tijdens-de-fotosessie-op-paleis-noordeinde.jpg
portretfoto-s/fotosessies/2022/zomer-2022/koninklijk-gezin-poseert-tijdens-de-fotosessie-op-paleis-noordeinde.jpg
portretfoto-s/fotosessies/2022/zomer-2022/prinses-amalia-prinses-alexia-en-prinses-ariane-tijdens-de-fotosessie-op-paleis-noordeinde.jpg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-01.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-02.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-03.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-04.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-05.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-06.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-07.jpeg
portretfoto-s/fotosessies/2023/winter-2023/fotosessie-paleis-huis-ten-bosch-08.jpeg
portretfoto-s/fotosessies/2023/zomer-2023/zomerfotosessie-2023-koning-willem-alexander-en-de-prinses-van-oranje.jpg
portretfoto-s/fotosessies/2023/zomer-2023/zomerfotosessie-2023-koning-willem-alexander-en-koningin-maxima.jpg
portretfoto-s/fotosessies/2023/zomer-2023/zomerfotosessie-2023-koninklijk-gezin-1.jpg
portretfoto-s/fotosessies/2023/zomer-2023/zomerfotosessie-2023-koninklijk-gezin-2.jpg
portretfoto-s/fotosessies/2024/herfst-2024/de-prinses-van-oranje-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koning-willem-alexander-en-hondje-mambo-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koning-willem-alexander-en-koningin-maxima-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koning-willem-alexander-en-prinses-van-oranje-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koningin-maxima-en-hondje-mambo-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koningin-maxima-prinses-van-oranje-prinses-alexia-en-prinses-ariane-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koninklijk-gezin-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/koninklijk-gezin-in-amsterdam-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/prinses-alexia-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/prinses-ariane-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/prinses-van-oranje-2024.jpeg
portretfoto-s/fotosessies/2024/herfst-2024/prinses-van-oranje-prinses-alexia-en-prinses-ariane-2024.jpeg
portretfoto-s/fotosessies/2024/zomer-2024/zomerfotosessie-2024-koning-en-koningin-maxima.jpg
portretfoto-s/fotosessies/2024/zomer-2024/zomerfotosessie-2024-koning-en-prinses-van-oranje.jpg
portretfoto-s/fotosessies/2024/zomer-2024/zomerfotosessie-2024-koninklijk-gezin-paleis-huis-ten-bosch.jpg
portretfoto-s/fotosessies/2024/zomer-2024/zomerfotosessie-2024-koninklijk-gezin.jpg
portretfoto-s/fotosessies/2025/zomer-2025/koning-en-koningin-maxima-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/koning-en-prinses-van-oranje-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/koning-willem-alexander-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/koninklijk-gezin-1-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/koninklijk-gezin-2-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/prinses-alexia-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/prinses-ariane-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/prinses-van-oranje-en-koningin-maxima-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/prinses-van-oranje-prinses-ariane-en-prinses-alexia-zomer-2025.jpeg
portretfoto-s/fotosessies/2025/zomer-2025/prinses-van-oranje-zomer-2025.jpeg
portretfoto-s/kerstkaart/2023/kerstfoto-2023.jpg
portretfoto-s/kerstkaart/2023/kerstkaart-2023.jpeg
portretfoto-s/kerstkaart/2024/foto-kerstkaart-2024.jpg
portretfoto-s/kerstkaart/2024/kerstkaart-2024.jpg
portretfoto-s/koning-willem-alexander/koning-willem-alexander-koningin-maxima-prinses-van-oranje-prinses-alexia-en-prinses-ariane---erwin-olaf---2018---liggend.jpg

'''

def download_file(url, dir_output):
    filename=url.split('/')[-1]
    if not filename.endswith('.jpg'):
        filename += '.jpg'
    path = os.path.join(dir_output, filename)
    try:
        if not os.path.exists(path):
            with requests.get(url, stream=True) as r:
                with open(path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
    except Exception as err:
        print(err)

def download_files(url_prefix, url_list, dir_output):
    n=url_list.count('\n')
    i=0
    for url in StringIO(url_list):
        url=url.rstrip()
        i+=1
        if url!='':
            print(i,'/',n,url.split('/')[-1])            
            download_file(url_prefix + url, dir_output)
    



# Recognize faces and show results ########################################################
#
# compare faces in a number of photos against a collection of known faces

def recognize_faces_show():    

    #all known persons
    known_faces=read_known_faces(dir_known_persons)

    plt.ion() #interactive mode to allow updating the plot while running
    plot_personsrow=[{'topleft':True}]
    for person in known_faces:
        for j,image in enumerate(known_faces[person]):
            plot_personsrow.append({'title':person+str(j), 'img':image['img']})

    #index all photos
    allimages = []
    for file in list(Path(dir_input).rglob('*.*')):
        if file.is_file and file.suffix.lower() in ['.jpg', '.jpeg']:
            allimages.append(file)

    n=len(allimages)
    log('allimages:',n)

    #random order, so it doesnt start with the same images with each run
    nindexes=np.random.choice(range(len(allimages)), size=n, replace=False)
    nimages=[ allimages[i] for  i in nindexes ]

    #compare
    mostfacesfile=None
    mostfaces=-1
    
    #all photos
    for k, file in enumerate(nimages):
        log('recognize',k,'/',n,file)

        #get faces from the photo
        nname=normalize_filename(dir_input,file)
        faces=deep_faces(file)

        #all faces in the photo
        result_total={}
        plot=[plot_personsrow]       
        for i,face in enumerate(faces['facesarray']):

            plot_row=[{'title':'face'+str(i), 'img':face['img_final']}]

            #all known persons
            maxconfidence=-1
            maxconfidenceid=None
            for p,person in enumerate(known_faces):

                #all known images of a person
                for j,faceknown in enumerate(known_faces[person]):
                    #compare
                    verifyresult = DeepFace.verify(faceknown['img'], face['img_final'], enforce_detection=False)
                    log('verify',person, j,'confidence',verifyresult['confidence']
                        ,'distance',verifyresult['distance'], 'threshold',verifyresult['threshold'])

                    #is it a match?
                    emphasize=0
                    cid=str(p)+','+str(j)
                    if verifyresult['confidence']>=recognize_minimum_confidence and verifyresult['distance'] <= verifyresult['threshold']:
                        emphasize=1
                        #keep track of best match
                        if verifyresult['confidence']>maxconfidence:
                            maxconfidenceid=cid
                            maxconfidence=verifyresult['confidence']
                        result_total[person]={
                            'face' : i,
                            'faceknown' : faceknown['name'],
                            'numfaces' : len(faces['facesarray']),
                            'facescore' : face['score'],
                            'verifyresult' : verifyresult
                            }
                            
                    label1='distance '+str(round(verifyresult['distance']*100))
                    label2='maxdist '+str(round(verifyresult['threshold']*100))
                    label3='match '+str(round(verifyresult['confidence']))+'%'
                    plot_row.append({'id':cid, 'label1':label1, 'label2':label2, 'label3':label3, 'emphasize':emphasize})

            #mark best match per face
            for col in plot_row:
                if 'id' in col and col['id']==maxconfidenceid:
                    col['emphasize']=2

            face['confidence']=maxconfidence

            plot.append(plot_row)

        #Note: one face might match multiple persons
        if len(result_total)>0:
            personsfoundstr=','.join(str(e) for e in result_total)
            filename=os.path.join(dir_output, personsfoundstr+'-'+nname)
            log('found',filename)

            #copy the file
            if debug>=2:
                shutil.copy(file, filename+'.jpg')

            #save-verify-result
            if debug>=1:
                with open(filename+'.py', 'w', encoding='utf-8') as f:
                    pprint({'w':faces['w'],'h':faces['h'],'file':str(file),
                            'results': result_total}, f, width=160, compact=True, sort_dicts=False)

            #Show the result
            close_all_windows()
            
            #show comparison result
            show_subplots(plot)

            #faces and features
            image=cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_UNCHANGED)
            show_faces_marked(image,faces)

