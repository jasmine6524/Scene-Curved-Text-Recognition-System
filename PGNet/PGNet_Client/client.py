import requests
import base64
import os

if __name__ == '__main__':

    '''
    img1 = f.read(open('./6.jpg', 'rb')) # img1->二进制数据
    img2 = base64.b64encode(img1)   # img2->对二进制数据进行base64编码，得到字节数据
    img3 = img2.decode() # img3->对字节类型数据进行base64解码得到字符串数据
    '''
    # url = 'http://192.168.2.20:5000/ocr'
    url = 'http://192.168.2.36:5000/ocr'
    imgs = os.listdir('./imgs')
    time = 0
    for img in imgs:
        img_path = os.path.join('./imgs', img)
        with open(img_path, 'rb') as f:
            img = base64.b64encode(f.read()).decode()
        file = {'image': img}
        r = requests.post(url, data=file)
        dic = r.json()
        time += float(dic['time'])
        print(img_path, dic['time'])
    print('Avg Time:{:.4f}'.format(time/len(imgs)))