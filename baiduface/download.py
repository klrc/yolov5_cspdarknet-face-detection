import base64
import requests
import os
import json
import time


def get_access_token():
    api_key = "EkOgWupeYEV8UUHoygGi4Fi2"
    secret_key = "39gKuCNF8KnOstC2lT7QsKlZ3hKwcbmN"
    host = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    response = requests.get(host)
    if response:
        access_token = response.json().get("access_token")
        return access_token
    else:
        raise Exception("response:" + response)


def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)
    return str(base64_data)[2:]


def request_ai(base64_data, access_token):
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
    face_field = "age,expression,face_shape,gender,glasses,landmark,quality,eye_status,emotion,face_type,mask,spoofing"
    params = '{"image":"' + base64_data + '","image_type":"BASE64","face_field":"' + face_field + '", "max_face_num":40}'
    request_url = request_url + "?access_token=" + access_token
    headers = {"content-type": "application/json"}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        return response.json()
    else:
        raise Exception("response:" + response)


if __name__ == "__main__":
    access_token = get_access_token()
    data_root = "/Users/sh/SourceCode/google_asian_old_people_0124/images"
    json_root = "/Users/sh/SourceCode/google_asian_old_people_0124/baidu_face_v3"
    ckpt = time.time()
    if not os.path.exists(json_root):
        os.makedirs(json_root)
    for image in os.listdir(data_root):
        if not image.startswith("._"):
            json_path = f"{json_root}/{image.split('.')[0]}.json"
            if os.path.exists(json_path):
                print(f"ignore: {json_path}")
                continue
            while True:
                try:
                    image_path = f"{data_root}/{image}"
                    base64_data = get_base64_image(image_path)
                    delta = time.time() - ckpt
                    if delta < 0.1:  # <=10 QPS
                        time.sleep(0.1 - delta)
                    json_data = request_ai(base64_data, access_token)
                    ckpt = time.time()
                    result = json_data.get("result")
                    face_num = 0
                    if result:
                        face_num = result.get("face_num")
                    print(image_path, "success", "face_num=", face_num)
                    with open(json_path, "w") as f:
                        json.dump(json_data, f)
                    break
                except Exception as e:
                    print(e)
                    continue
