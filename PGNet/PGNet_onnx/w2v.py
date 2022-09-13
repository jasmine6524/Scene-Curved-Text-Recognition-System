from aip import AipSpeech

def str2Talk(inputWords,volum=5,speed=5,pitch=5,person=4):
    """ 你的 APPID AK SK """
    APP_ID = 'xxxxxxxx'
    API_KEY = 'xxxxxxxx'
    SECRET_KEY = 'xxxxxxxx'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    stringM= inputWords

    result  = client.synthesis(stringM, 'zh', 1, {
        'vol': volum,   #音量，取值0-9，默认为5中语速
        'spd': speed,   #语速，取值0-9，默认为5中语速
        'pit': pitch,   #音调，取值0-9，默认为5中语调
        'per': person,  #度小美=0(默认)，度小宇=1，，度逍遥（基础）=3，度丫丫=4
        
        
    })

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        print("成功")
        with open('./results/res.mp3', 'wb') as f:
            f.write(result)
    else:
        print(result)

if __name__ == '__main__':
    userInput = input("input words: ")
    str2Talk(userInput,5,5,5,4)
