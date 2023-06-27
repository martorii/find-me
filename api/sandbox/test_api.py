import requests


def test_face_distance():
    url = 'http://127.0.0.1:8000/get_face_distance'
    files = [
        ('files', open('../cv/similarity_model/data/own/face_christina.png', 'rb')),
        ('files', open('../cv/similarity_model/data/own/face_christina_2.png', 'rb'))
    ]
    resp = requests.get(url=url, files=files)
    print(resp.json())


def test_api_find_face_in_picture():
    url = 'http://127.0.0.1:8000/find_face_in_picture'
    files = [
        ('files', open('../cv/similarity_model/data/own/face_sophia_2.png', 'rb')),
        ('files', open('../cv/similarity_model/data/own/img_multi_boda_2.jpg', 'rb'))
    ]
    resp = requests.get(url=url, files=files)
    print(resp.json())

def test_api_filter_pictures_with_face():
    url = 'http://127.0.0.1:8000/filter_pictures_with_face'
    files = [
        ('files', open('../cv/similarity_model/data/own/face_sophia_2.png', 'rb')),
        ('files', open('../cv/similarity_model/data/own/img_multi_boda_2.jpg', 'rb'))
    ]
    resp = requests.get(url=url, files=files)
    print(resp.json())

# test_face_distance()

# test_api_find_face_in_picture()

test_api_filter_pictures_with_face()

