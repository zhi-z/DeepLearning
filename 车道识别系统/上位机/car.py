
# coding: utf-8

# 道路预测与小车控制

# import some package.
import numpy as np
import cv2
from scipy.misc import imresize

from keras.models import load_model
from my_uart import MyUart
import time


# 1. 小车控制

# 创建小车控制类
class Car():
   # 标志作用
    left_flag = 0   
    right_flag = 0
    

    def start(self,x):
        """
        开始:小车初始化
            也可以当做静止
        """
        if x.is_open:
            data = '005'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
            print('开始')
        else:
             print('串口未打开，请打开串口发送数据。')  
    
    def stop(self,x):
        """
            停止
        """
        if x.is_open:
            data = '006'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
            print('停止')
        else:
             print('串口未打开，请打开串口发送数据。')               
               
                
    def car_forward(self,x):
        """
            前进
        """
        if x.is_open:
            data = '012'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
            print('前进')
        else:
             print('串口未打开，请打开串口发送数据。')   
                
                
    def car_back(self,x):
        """
            后退
        """
        if x.is_open:
            data = '015'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
            print('后退')
        else:
             print('串口未打开，请打开串口发送数据。')         
        
    def car_left(self,x):
        """
            左转
        """
        if x.is_open:
            data = '013'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
            print('左转')
        else:
             print('串口未打开，请打开串口发送数据。')        
    
    def car_right(self,x):
        """
            右转
        """
        if x.is_open:
            data = '014'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
            print('右转')
        else:
             print('串口未打开，请打开串口发送数据。')     
    
    def left_speed(self,x,value):
        """
            左轮速度滑动条
        """
        if x.is_open:
            data = str(value)
            if len(data) == 1:
                data = '0'+'0'+data
            elif len(data) == 2:
                data = '0'+data
            else:
                pass
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            x.write(demo) 
        else:
            print('串口未打开，请打开串口发送数据。')
        
    def right_speed(self,x,value):
        """
        右轮速度滑动条，这里在要value = 260~460，为了显示的时候以50~250为准，所以要减去210。
        """
        value = value + 210
        if x.is_open:
            data = str(value)
            if len(data) == 1:
                data = '0'+'0'+data
            elif len(data) == 2:
                data = '0'+data
            else:
                pass
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            # 通过串口发送数据
            x.write(demo)   
        else:
            print('串口未打开，请打开串口发送数据。')   
            
    def car_run(self,car,x,p_1,p_2,error_left = 70,error_right = 50,
				v_left = 120,v_right = 190,forward_left = 143,forward_right = 143):
        
        if p_1 > error_right:          
            self.left_flag = 0
            self.right_flag = 0
            # youzhuann
            car.right_speed(x,v_right)
            car.left_speed(x,v_left)

        elif p_1 < - error_left:           
            self.left_flag = 0
            self.right_flag = 0
            # zuozhuan
            car.right_speed(x,v_left)
            car.left_speed(x,v_right)
        else:
            # 向前
            car.right_speed(x,forward_right)
            car.left_speed(x,forward_left)     


# 2. 预测部分


# 道路预测
class RoadPre():
	
    # 获取坐标函数
    def coordinate(self,image,y_star,y_stop,x_size):
        p_x1_lis = []
        p_y1_lis = []
        for y in range(y_star,y_stop):
            for x in range(x_size):
                if image[y][x] != 0:
                    p_x1_lis.append(x)
                    p_y1_lis.append(y)

        if len(p_x1_lis)==0 or len(p_y1_lis)==0:
            pass
        else:
            p_x1 = np.mean(p_x1_lis)
            p_y1 = np.mean(p_y1_lis)
            p = (int(p_x1),int(p_y1))
            return p    
    # 绘制线条
    def draw_line(self,car,uart,image_g,error_left = 70,error_right = 50,
				  v_left = 120,v_right = 190,y_bottom = 250,y_top = 300,forward_left = 143,forward_right = 143):
        # 获取两个坐标
        p_1 = self.coordinate(image_g,y_bottom,y_top,640)
        p_2 = self.coordinate(image_g,475,480,640)
        if p_1 != None and p_2 != None:
            car.car_run(car,uart,p_1[0] - 320,p_2,error_left,error_right,v_left,v_right,forward_left,forward_right)
            print('p1:',p_1[0]-320)
        else:
            pass
        image_g = np.array(image_g,np.uint8) 
#         print(image_g.shape)
        n = cv2.line(image_g,p_1,p_2,0,10)

        return image_g

    def run_pre(self,car,uart,error_left = 70,error_right = 50,v_left = 120,v_right = 190,
				y_bottom = 250,y_top = 300,forward_left = 143,forward_right = 143,
				model_name = 'final_test_model.h5',star=1):
        
        model = load_model(model_name) 
        # videoCapture = cv2.VideoCapture("./input/pre.mp4")
        videoCapture = cv2.VideoCapture(0)
         #读帧
        success, frame = videoCapture.read()
        
        while success:
            small_img = imresize(frame, (80, 160, 3))
            small_img = np.array(small_img)
            # return (1,80, 160, 1)
            prediction = model.predict(small_img.reshape(1,80, 160, 3))
            prediction = prediction.reshape(80, 160, 1)

            blanks = np.zeros_like(prediction).astype(np.uint8)
            lane_drawn = np.dstack((blanks, prediction, blanks))

            lane_image = imresize(lane_drawn, (480, 640, 3))
            # 绘制直线，控制速度等
            re_line = self.draw_line(car,uart,lane_image[:,:,1],error_left,error_right,
									v_left,v_right,y_bottom,y_top,forward_left,forward_right)
            
            lane_image[:,:,1] = re_line
 
            result = cv2.addWeighted(frame, 1, lane_image, 1, 0)   

            cv2.imshow("result",result)
            success, frame = videoCapture.read() 
			
				
	
            if cv2.waitKey(100) & 0xFF == ord('q') or star != 1:
                videoCapture.release()
                cv2.destroyAllWindows()    

                return    

