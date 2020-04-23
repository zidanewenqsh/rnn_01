'''
@Descripttion: 
@version: 
@Author: QsWen
@Date: 2020-04-22 20:21:05
@LastEditors: QsWen
@LastEditTime: 2020-04-22 20:32:28
'''
import random
import os
from PIL import Image,ImageDraw,ImageFont,ImageFilter

def randChar():
    return chr(random.randint(48,57))

def randBgColor():
    return (
        random.randint(50,100),
        random.randint(50,100),
        random.randint(50,100)
    )

def randTestColor():
    return (
        random.randint(90,160),
        random.randint(90,160),
        random.randint(90,160)
    )

w = 30*4
h = 60
font = ImageFont.truetype("arial.ttf",size=36)

for i in range(1000):
    image = Image.new(mode='RGB',size=(w,h),color=(0,0,0))
    draw = ImageDraw.Draw(image)
    for x in range(w):
        for y in range(h):
            draw.point((x,y),fill=randBgColor())
    filename = []
    for t in range(4):
        ch = randChar()
        filename.append(ch)
        draw.text((30*t+random.randint(0,5),10+random.randint(-2,2)),ch,font=font,fill=randTestColor())
    image = image.filter(ImageFilter.BLUR)
    image_path = r"../datas/captcha_img_test"
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    # print("".join(filename))
    image.save("{0}/{1}.jpg".format(image_path,"".join(filename)))
    # print(i)
