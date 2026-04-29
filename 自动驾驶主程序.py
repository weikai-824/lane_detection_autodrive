import hqyj_mqtt
import cv2
import numpy as np
import base64
import queue
import matplotlib.pyplot as plt
import time
from pid import PID
import json
class HelpSeePicture:
    def __init__(self,maxframe=200,image_height=480):
        plt.ion()
        #初始化图表和坐标轴，和曲线
        self.fig,self.ax=plt.subplots()
        self.lane_center, = plt.plot([],[],'r-',label='lane_center')
        self.image_center, = plt.plot([], [], 'b-', label='image_center')
        #初始化标题和标签
        self.ax.set_title('lane and image center')
        self.ax.set_xlabel('frame')
        self.ax.set_ylabel('pixel coordinates')
        self.ax.legend()
        #初始化数据
        self.x_data=[]
        self.y_lane_data=[]
        self.y_image_data=[]
        self.maxframe=maxframe
        self.maxheight=image_height
        #设置初始化函数
        self.init_plot()
    def init_plot(self):
        self.ax.set_xlim(0,self.maxframe)
        self.ax.set_ylim(0,self.maxheight)
        self.lane_center.set_data([],[])
        self.image_center.set_data([],[])
        self.ax.grid()
    def update(self,frame,lane_center,image_center):
        #将值传进初始化的空列表里
        self.x_data.append(frame)
        self.y_lane_data.append(lane_center)
        self.y_image_data.append(image_center)
        #更新折线图
        self.lane_center.set_data(self.x_data,self.y_lane_data)
        self.image_center.set_data(self.x_data, self.y_image_data)
        #保持x轴范围稳定
        if len(self.x_data) > self.maxframe:
            self.ax.set_xlim(self.x_data[-self.maxframe],self.x_data[-1])
            self.ax.figure.canvas.draw()
        plt.pause(0.01)
def b64tocv(msg):

    #将字典里的值转换为字节
    img_byt=base64.b64decode(msg['image'])
    # print(img_byt)

    #将字节转换为一维数组
    img_nu=np.frombuffer(img_byt,dtype=np.uint8)#uint8是0-255的无符号整数
    # print(img_nu)

    #将这个一维数组转换为三通道图像
    img_cv=cv2.imdecode(img_nu,cv2.IMREAD_COLOR)
    return img_cv

def perspective_transform(image_ori):
    image_shape=np.shape(image_ori)
    h,w=image_shape[:2]

    # cv2.line(image_ori,(38,h),(w//2-37,h//2-20),(0,0,255),1)
    # cv2.line(image_ori,(428,h),(w//2+42,h//2-20),(0,0,255),1)
    src=np.float32([[35,h],[420,h],[w//2+38,h//2-20],[w//2-40,h//2-20]])
    des=np.float32([[w/4,h],[w *3/4 ,h],[w*3/4 ,0],[w/4,0]])

    #透射变换矩阵
    M=cv2.getPerspectiveTransform(src,des)

    #逆透射变换矩阵
    miv=cv2.getPerspectiveTransform(des,src)

    image_des=cv2.warpPerspective(image_ori,M,(w,h))
    return image_des,miv

def get_line_tidu(image_wrap):
    #对传进来的图像先初步进行一次滤波去除噪声
    image_gas=cv2.GaussianBlur(image_wrap,(5,5),1)
    #梯度处理需要灰度图
    image_gray=cv2.cvtColor(image_gas,cv2.COLOR_BGR2GRAY)
    #梯度处理
    image_ti=cv2.Sobel(image_gray,-1,1,0)
    #用二值化处理不想要的边界和噪声
    ret,image_bin=cv2.threshold(image_ti,127,255,cv2.THRESH_BINARY)
    #闭运算
    image_increase=dilate_erode(image_bin)
    #进行了一次中值滤波去掉孤立小白点
    image_med=cv2.medianBlur(image_increase,9)
    return image_med

def get_line_white(image,thresh=(220,255)):
    image_hls=cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    channel_l=image_hls[:,:,1]
    minval,maxval,minloc,maxloc=cv2.minMaxLoc(channel_l)
    channel_l=((channel_l-minval)/(maxval-minval))*255
    res_l=np.zeros_like(channel_l)
    res_l[(channel_l > thresh[0])&(channel_l <=thresh[1])]=1
    return res_l

def get_line_yellow(image,thresh=(170,220)):
    #提取黄色车道线时，我只要左边一半的图像防止干扰
    image[:,240:,:]=0
    image_lab=cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
    channel_b=image_lab[:,:,2]
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(channel_b)
    if maxval >100:
        channel_b=((channel_b-minval)/(maxval-minval))*255
        res_b=np.zeros_like(channel_b)
        res_b[(channel_b > thresh[0])&(channel_b <=thresh[1])]=1
        return res_b

def get_line_color(image_wrap_copy):
    image_wrap_copy=cv2.GaussianBlur(image_wrap_copy,(3,3),1)
    #提取白色车道线，用HLS颜色模式
    res_l=get_line_white(image_wrap_copy)
    #提取黄色车道线，用Lab颜色模式
    res_b=get_line_yellow(image_wrap_copy)
    #图像融合
    res=cv2.add(res_b,res_l)
    #进行图像增强
    res_increase=dilate_erode(res)
    return res_increase

def dilate_erode(image_bin):
    #有可能得到的车道线会有断开，做一次闭运算，数据增强
    kernal_1=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    kernal_2=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    image_dil=cv2.dilate(image_bin,kernal_1,iterations=1)
    image_ero=cv2.erode(image_dil,kernal_2,iterations=1)
    return image_ero

def findingline(image_tidu):
    h, w = image_tidu.shape[:2]
    # 直方图找起点，为避免峰值白色像素点没有出现在最底部，适当缩小找的范围只找图像四分之一即可
    hisdiagram = np.sum(image_tidu[h * 1 // 2:, :], axis=0)
    # plt.plot(hisdiagram)
    # plt.show()
    minlocal = hisdiagram.shape[0] // 2
    leftx_base = np.argmax(hisdiagram[:minlocal])
    right_base = np.argmax(hisdiagram[minlocal:]) + minlocal
    # print(leftx_base)
    # print(right_base)
    # 由于传进来的图像是单通道不方便后续画窗口方框，所以此时叠加成三通道的
    org_img = np.dstack((image_tidu, image_tidu, image_tidu))
    # 找到图中所有的白色像素点索引/非零像素点
    nonzero = image_tidu.nonzero()
    # print(nonzero)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 设置窗口参数
    windows = 10
    height = h // windows
    halfwidth = 55
    minpix = 50
    # 初始化
    leftx_current = leftx_base
    rightx_current = right_base
    leftx_pre=leftx_current
    right_pre=rightx_current
    # 初始化空列表用于接收左右两车道线的白色像素点的位置索引
    left_white = []
    right_white = []
    # 滑动窗口主循环
    for window in range(windows):
        # 计算当前窗口位置
        # 上下边界
        win_high_y = int(h - height * (window + 1))
        win_low_y = int(h - height * window)
        # 左车道线
        win_left_xleft = leftx_current  - halfwidth
        win_right_xleft = leftx_current  + halfwidth
        # 右车道线
        win_left_xright = rightx_current - halfwidth
        win_right_xright = rightx_current + halfwidth
        #画出来看一看
        cv2.rectangle(org_img,(win_left_xleft,win_high_y),(win_right_xleft,win_low_y),(0,255,0),2)
        cv2.rectangle(org_img, (win_left_xright, win_high_y), (win_right_xright, win_low_y), (0, 255, 0), 2)
        # 找到处于窗口内所有白色像素点坐标的条件
        condiony = (nonzeroy >= win_high_y) & (nonzeroy < win_low_y)
        condionxleft = (nonzerox >= win_left_xleft) & (nonzerox <= win_right_xleft)
        condionxright = (nonzerox >= win_left_xright) & (nonzerox <= win_right_xright)
        # 这两个是布尔数组
        final_condion_left = condiony & condionxleft
        final_condion_right = condiony & condionxright
        # 找到窗口内白色像素点的位置索引(同时满足x和y的)
        good_left = final_condion_left.nonzero()[0]
        good_right = final_condion_right.nonzero()[0]
        left_white.append(good_left)
        right_white.append(good_right)
        #开始更新小窗口位置(滑动窗口)
        if len(good_left) > minpix:
            leftx_current=int(np.mean(nonzerox[good_left]))
        else:
            if len(good_right) > minpix:
                offset=int(np.mean(nonzerox[good_right]))-right_pre
                leftx_current=leftx_current+offset
        if len(good_right) > minpix:
            rightx_current=int(np.mean(nonzerox[good_right]))
        else:
            if len(good_left) > minpix:
                offset=int(np.mean(nonzerox[good_left])-leftx_pre)
                rightx_current=rightx_current+offset
        #记录上一次位置
        leftx_pre=leftx_current
        right_pre=rightx_current
    #由于之前的left_white是列表的列表，需要合并
    left_white=np.concatenate(left_white)
    right_white=np.concatenate(right_white)
    if len(left_white) < 10 or len(right_white) < 10:
        return None,None,None,None
    else:
        #提取实际的车道线坐标
        leftx=nonzerox[left_white]
        lefty=nonzeroy[left_white]
        rightx=nonzerox[right_white]
        righty=nonzeroy[right_white]
        #多项式拟合,绘制一条平滑曲线,里面存放的是a,b,c三个参数
        leftfit=np.polyfit(lefty,leftx,2)
        rightfit=np.polyfit(righty,rightx,2)
        #找到垂直方向所有y坐标
        y=np.linspace(0,h-1,h)
        #再找到车道线对应所有x坐标
        leftx_fit=leftfit[0] * y**2 + leftfit[1] * y + leftfit[2]
        rightx_fit = rightfit[0] * y ** 2 + rightfit[1] * y + rightfit[2]
        # org_img[lefty,leftx]=[0,0,255]
        # org_img[righty,rightx]=[255,0,0]
        #计算中心车道线位置
        middlelx_fit=(leftx_fit+rightx_fit)/2
        # cv2.imshow('org_img', org_img)
        # cv2.waitKey(0)
        return leftx_fit,rightx_fit,middlelx_fit,y

def show_line(image_ori,image_warp,image_tidu,miv,leftx_fit,rightx_fit,middlelx_fit,y):
    #需要原图，鸟瞰图，叠加效果图，纯车道线图
    #创建一张鸟瞰空间的空白画布
    zero_img=np.zeros_like(image_tidu).astype(np.uint8)
    color_img=np.dstack((zero_img,zero_img,zero_img))
    h,w=color_img.shape[:2]
    #合并坐标
    points_left=np.transpose(np.vstack((leftx_fit,y)))
    points_right=np.transpose(np.vstack((rightx_fit,y)))
    points_middle=np.transpose(np.vstack((middlelx_fit,y)))
    # print(points_right.shape)
    #在空白画布上画车道线
    cv2.polylines(color_img,np.int32([points_left]),isClosed=False,color=(202,124,0),thickness=13)
    cv2.polylines(color_img, np.int32([points_right]), isClosed=False, color=(202, 124, 0), thickness=13)
    cv2.polylines(color_img, np.int32([points_middle]), isClosed=False, color=(202, 124, 0), thickness=13)
    # cv2.imshow('zz',color_img)
    # cv2.waitKey(0)
    #将鸟瞰图下的车道线逆变换成原图视角下的
    new_img=cv2.warpPerspective(color_img,miv,(w,h))
    # cv2.imshow('zz',new_img)
    # cv2.waitKey(0)
    #将逆变换后的图像和原图像进行加权融合,得到叠加效果图
    add_img=cv2.addWeighted(image_ori,1,new_img,1,0)
    #创建纯灰色背景图用来显示车道线
    zero_img1=np.zeros_like(image_tidu).astype(np.uint8) + 127
    color_img1=np.dstack((zero_img1,zero_img1,zero_img1))
    add_img1=cv2.addWeighted(color_img1,1,new_img,1,0)
    # cv2.imshow('image_ori',image_ori)
    # cv2.imshow('image_warp',image_warp)
    # cv2.imshow('add_img', add_img)
    # cv2.imshow('add_img1', add_img1)ww
    resu1=np.concatenate((image_ori,image_warp),axis=1)
    resu2=np.concatenate((add_img,add_img1),axis=1)
    resu=np.concatenate((resu1,resu2),axis=0)
    cv2.imshow('resu', resu)

    cv2.waitKey(1)
    return None
def anto_drive(image_ori,middlelx_fit,pid,mqtt_client):
    # 车道线平均位置，应该是目标位置但因为总是在变化我们把他当作当前位置
    lane_center = middlelx_fit[210:].mean()
    # 当前车的位置，但这个是不变的所以我们姑且当作目标位置
    image_center = image_ori.shape[1] // 2
    change_angle = -pid(lane_center)
    # 根据直线或弯道改变速度
    angle_abs = abs(change_angle)
    if angle_abs < 3:  # 近似直线
        carSpeed = 23
    elif angle_abs < 6:  # 缓弯
        carSpeed = 16
    elif angle_abs < 10:  # 中等弯
        carSpeed = 10
    else:  # 急弯
        carSpeed = 8
    # 发送指令
    mqtt_client.send_mqtt(json.dumps({'carSpeed': carSpeed}))
    mqtt_client.send_mqtt(json.dumps({'carDirection': change_angle}))
    # print(change_angle)
    return lane_center,image_center
if __name__ == '__main__':
    last_leftx_fit = None
    last_rightx_fit = None
    last_middlelx_fit = None
    last_y = None
    lost_frame_count = 0  # 连续丢失帧计数
    MAX_LOST_FRAME = 5
    #构建一个队列，用来储存传入的3D场景里的数剧
    queue_mqtt=queue.Queue(5)
    # start_time=time.time()
    # current_pic=0
    #初始化文件名
    # i=1
    frame=0
    plotter=HelpSeePicture()
    pid=PID(0.25,0.00,0.00,setpoint=240,sample_time=0.128,output_limits=(-13,13))
    #构建一个mqtt客户端,并与mqtt服务器连接，便于与后面3D场景通信
    mqtt_client=hqyj_mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',queue_mqtt)
    try:
        while True:
        # #     #1.接收数据
            msg=queue_mqtt.get()
        # #     #此时给的是json解析后的字典
        # #     # print(msg)
        # #     # #判断image这个键在不在msg这个字典里，判断传进来的是不是图像数据
            if 'image' in msg:
        #         # current_pic+=1
        #         # current_time=time.time()
        #         # inter_time=current_time-start_time
        #         # fps=current_pic / inter_time
        #         # print(fps)
                image_ori=b64tocv(msg)
        #         image_ori=cv2.imread('./7.png')

                #2.进行透视变换
                image_warp,miv=perspective_transform(image_ori)
                image_warp_copy=image_warp.copy()
        #         #
        #         3.提取车道线
        #         第一种，梯度提取,注意这提取的只是有可能的车道线，随着视角变换可能引进其他车道线
                image_tidu=get_line_tidu(image_warp)

                # 第二种，颜色提取
                # image_color=get_line_color(image_warp_copy)

                #4.车道线拟合
                leftx_fit,rightx_fit,middlelx_fit,y=findingline(image_tidu)
                if leftx_fit is None or rightx_fit is None:
                    lost_frame_count += 1
                    if lost_frame_count <= MAX_LOST_FRAME and last_middlelx_fit is not None:
                        # 使用上一帧的车道线数据（假设车道线不会突变）
                        leftx_fit = last_leftx_fit
                        rightx_fit = last_rightx_fit
                        middlelx_fit = last_middlelx_fit
                        y = last_y
                        print(f"车道线丢失，使用上一帧数据 (丢失次数: {lost_frame_count})")
                    else:
                        # 严重丢失：保持车辆减速并直行，同时报警
                        mqtt_client.send_mqtt(json.dumps({'carSpeed': 8}))
                        mqtt_client.send_mqtt(json.dumps({'carDirection': 0}))
                        print("严重丢失车道线，降级为慢速直行")
                        continue  # 这次不进行显示、绘图，但已发送安全指令
                else:
                    # 检测成功，更新缓存和丢失计数
                    lost_frame_count = 0
                    last_leftx_fit = leftx_fit
                    last_rightx_fit = rightx_fit
                    last_middlelx_fit = middlelx_fit
                    last_y = y

                #5.车道线的显示
                show_line(image_ori,image_warp,image_tidu,miv,leftx_fit,rightx_fit,middlelx_fit,y)

                #6.自动驾驶
                lane_center , image_center=anto_drive(image_ori,middlelx_fit,pid,mqtt_client)
        #
                #7.绘制辅助折线图
                plotter.update(frame,lane_center,image_center)
                frame+=1
                # print(frame)
                # cv2.imshow('image_ori', image_ori)
                # cv2.imshow('image_warp', image_warp)
                # cv2.imshow('image_tidu', image_tidu)
                # cv2.imshow('image_color', image_color)
                # cv2.waitKey(0)
    except Exception as e:
        print(e)
