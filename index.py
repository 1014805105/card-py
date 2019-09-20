import tkinter as  tk
from  tkinter import *
from  tkinter import filedialog

def topframe():

    def pathf():
        Fpath = filedialog.askopenfilename()
        path.set(Fpath)
        f = open('lujing.txt', 'w')
        f.write(str("".join(Fpath)))
        f.close()
        print(Fpath)
    def tuichu():
        clientmain.destroy()




    card=tk.Toplevel()
    path=StringVar()
    card.title('银行卡选择')
    card.geometry('300x100')

    l1=tk.Label(card,text='选择银行卡图片',font=("Arial", 12))
    l2=tk.Label(card,text='目标路径').grid(row =1,column = 1)
    E=tk.Entry(card,textvariable =path).grid(row =1,column = 2)
    b=tk.Button(card,text='路径选择',command= pathf).grid(row =1,column = 3)
    b1=tk.Button(card,text='submit',command= tuichu).grid(row=2,column=2)
    l1.grid(row =0,column = 2)
def webindex():
    import webbrowser
    url = 'D:/py/template-matching-ocr/web/default.html'
    webbrowser.open_new_tab(url)

clientmain=tk.Tk()
l1=tk.Label(clientmain,text='银行卡挂失系统',font=("Arial", 20)).grid(row=0,column=2)
clientmain.image = tk.PhotoImage(file='4E2.png')
clientmain.image2=tk.PhotoImage(file='IE2.png')
clientmain.image3=tk.PhotoImage(file='exit2.png')
BS=Button(clientmain,text='选择银行卡图片',command=topframe,
		  image=clientmain.image).grid(row=1,column=1)

Bweb=Button(clientmain,text='网页版',command=webindex, image=clientmain.image2).grid(row=1,column=2)
BE=Button(clientmain,text="退出",command=quit,image=clientmain.image3).grid(row=1,column=3)
clientmain.geometry("642x300")
clientmain.mainloop()

from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

f=open("lujing.txt","r")
lujing=f.read()
print(lujing)
f.close()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,default=lujing,
				help="path to input image")
ap.add_argument("-r", "--reference", default=" ocr_a_reference.png", required=False,
				help="path to reference OCR-A image")
args = vars(ap.parse_args())

# 定义信用卡类型
FIRST_NUMBER = {
	"3": "美国运通组织",
	"4": "Visa",
	"5": "万事达组织",
	"6": "银联组织"
}

# 通过加载参考OCR-A图像开始图像处理
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# 在OCR-A字体图像上找到轮廓
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
						   cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 循环浏览轮廓，提取ROI并将其与相应的数字相关联
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi

# 找到并隔离数字，启动模板匹配来识别每个数字
# 初始化几个构造核函数的结构
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 准备去OCR的图像
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
				  ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
					   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	if ar > 2.5 and ar < 4.0:

		if (w > 40 and w < 55) and (h > 10 and h < 20):
			locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])
output = []

for (i, (gX, gY, gW, gH)) in enumerate(locs):

	groupOutput = []

	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	group = cv2.threshold(group, 0, 255,
						  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
								 cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = imutils.grab_contours(digitCnts)
	digitCnts = contours.sort_contours(digitCnts,
									   method="left-to-right")[0]

	for c in digitCnts:

		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))

		scores = []

		for (digit, digitROI) in digits.items():
			result = cv2.matchTemplate(roi, digitROI,
									   cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		groupOutput.append(str(np.argmax(scores)))

	cv2.rectangle(image, (gX - 5, gY - 5),
				  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	output.extend(groupOutput)

print("银行所属: {}".format(FIRST_NUMBER[output[0]]))
print("银行卡号 #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.imwrite("Image.png",image)
cv2.waitKey(0)

f = open('number.txt','w')
f.write(str("".join(output)))
f.close()
f=open('suoshu.txt','w')
f.write(str(FIRST_NUMBER[output[0]]))
f.close()
show=tk.Tk()
Ls=Label(show,text="银行卡号为 #: {}".format("".join(output))).grid(row=1,column=1)
ls=Label(show,text="银行所属为: {}".format(FIRST_NUMBER[output[0]])).grid(row=2,column=1)
BE1=Button(show,text="确定",command=show.destroy).grid(row=3,column=1)

show.mainloop()
clientmain.mainloop()
kahao="".join(output)
suoshu=FIRST_NUMBER[output[0]]
import csv
import codecs
fileName = 'list.csv'
with codecs.open(fileName, 'w', 'utf-8') as csvfile:
    filednames = ['卡号', '所属']
    writer = csv.DictWriter(csvfile, fieldnames=filednames)
    lists=[]
    list = {
        'number':kahao,
        'belong':suoshu,
    }
    lists.append(list)

    writer.writeheader()
    for list in lists:
        try:
            writer.writerow({'卡号': list['number'], '所属': list['belong']})
        except UnicodeEncodeError:
            print("编码错误, 该数据无法写到文件中, 直接忽略该数据")
import time
time.sleep(0.2)

import os
os.system("python index.py")