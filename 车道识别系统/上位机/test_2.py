# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

from Ui_test_2 import Ui_MainWindow
from my_uart import MyUart
import serial.tools.list_ports
from car import Car, RoadPre
import cv2
import time
 
    
    
class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.comboBox.setItemText(0, '') 
        com = MyUart().gerSerialName()
        self.comboBox.setItemText(0, com)
    
    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        清空多行文本框函数
        """
        # TODO: not implemented yet
        self.textBrowser.setHtml('')
    
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        """
        开始
        """
        if x.is_open:
            # 禁用自动模式
            self.pushButton_2.setDisabled(True) 
            self.pushButton_12.setDisabled(True) 
            
            data = '005'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            self.textBrowser.append('开始')
            print('开始')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')           
    
    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        """
            停止
        """
        if x.is_open:
            # 启动自动模式按键
            self.pushButton_2.setDisabled(False) 
            self.pushButton_12.setDisabled(False) 
            
            data = '006'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            self.textBrowser.append('停止')
            print('停止')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')           
             
    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        """
        Slot documentation goes here.
        前进
        """
        if x.is_open:
            data = '012'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            self.textBrowser.append('前进')
            print('前进')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')           

    
    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        """
        Slot documentation goes here.
        左转
        """
        if x.is_open:
            data = '013'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            self.textBrowser.append('左转')
            print('左转')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')       
    
    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        """
        Slot documentation goes here.
        右转
        """
        if x.is_open:
            data = '014'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            self.textBrowser.append('右转')
            print('右转')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')       
    
    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        """
        Slot documentation goes here.
        后退
        """
        if x.is_open:
            data = '015'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            self.textBrowser.append('后退')
            print('后退')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')       
        
    
    @pyqtSlot(str)
    def on_comboBox_activated(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type str
        串口选择
        """
        x.port = p0     #端口是COM3 
        print('串口选择', p0)
        self.textBrowser.append('选择端口：')
        self.textBrowser.append(p0)
        
    
    @pyqtSlot(int)
    def on_horizontalSlider_valueChanged(self, value):
        """
        Slot documentation goes here.
        
        @param value DESCRIPTION
        @type int
        50(反转最快)，100（反转中等速度），150(停止)，200(正传中等速度)，250(正转最快速度)
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
            print(demo)
            x.write(demo) 
            self.lcdNumber.display(value)
        else:
            print('串口未打开，请打开串口发送数据。')
            self.textBrowser.append('串口未打开，请打开串口发送数据。')
            
    @pyqtSlot(int)
    def on_horizontalSlider_2_valueChanged(self, value):
        """
        Slot documentation goes here.
        右轮速度滑动条，这里在要value = 260~460，为了显示的时候以50~250为准，所以要减去210。
        50(反转最快)，100（反转中等速度），150(停止)，200(正传中等速度)，250(正转最快速度)
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
            print(value-210)
            # 通过串口发送数据
            x.write(demo)   
            self.lcdNumber_2.display(value-210)
        else:
            print('串口未打开，请打开串口发送数据。')
            self.textBrowser.append('串口未打开，请打开串口发送数据。')       
  
  
    @pyqtSlot()
    def on_pushButton_9_clicked(self):
        """
        Slot documentation goes here.
        刷新
        """
        com_name = MyUart().gerSerialName()
        self.comboBox.setItemText(0, com_name)
        print('刷新:', com_name)
        
      

    @pyqtSlot()
    def on_pushButton_10_clicked(self):
        """
        Slot documentation goes here.
        连接串口
        """
        if x.is_open:
            print('串口已经打开')
            self.textBrowser.append('串口已经打开')
        else:
            x.open()
            self.textBrowser.append('打开串口')
            print('打开串口')

        
    
    @pyqtSlot()
    def on_pushButton_11_clicked(self):
        """
        Slot documentation goes here.
        关闭串口
        """
        print('串口关闭')
        x.close()
        self.textBrowser.append('关闭串口')

# 打开串口
    
    @pyqtSlot()
    def on_pushButton_12_clicked(self):
        """
        自动模式停止
        """
        if x.is_open:
            # 允许手动模式
            self.horizontalSlider.setDisabled(False)
            self.horizontalSlider_2.setDisabled(False)
            self.pushButton_3.setDisabled(False) 
            self.pushButton_4.setDisabled(False) 
            self.pushButton_5.setDisabled(False) 
            self.pushButton_6.setDisabled(False) 
            self.pushButton_7.setDisabled(False) 
            self.pushButton_8.setDisabled(False) 
            
            # 关闭道路图像窗口 & 各个速度为0
            roadPre.run_pre(car,x, v_left = 156,v_right = 156, 
                    forward_left = 156,forward_right = 156, star=0)
                    
            
            data = '006'
            demo = data.encode('utf-8')#由于串口使用的是字节，故而要进行转码，否则串口会不识别   
            print(demo)
            x.write(demo) 
            car.start(x)
            self.textBrowser.append('停止')
            print('停止')
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')       
        
    
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        自动行驶开始。
        """
        if x.is_open:
            # 禁用手动模式
            self.horizontalSlider.setDisabled(True)
            self.horizontalSlider_2.setDisabled(True)
            self.pushButton_3.setDisabled(True) 
            self.pushButton_4.setDisabled(True) 
            self.pushButton_5.setDisabled(True) 
            self.pushButton_6.setDisabled(True) 
            self.pushButton_7.setDisabled(True) 
            self.pushButton_8.setDisabled(True) 
            
            car.start(x)
            self.textBrowser.append('前进速度 左：'+str(self.spinBox.value())+',右：'+str(self.spinBox_2.value()))
            self.textBrowser.append('转向速度 左：'+str(self.spinBox_4.value())+',右：'+str(self.spinBox_3.value()))
            self.textBrowser.append('直行范围 左：'+str(self.spinBox_5.value())+',右：'+str(self.spinBox_6.value()))
            self.textBrowser.append('观察范围 左：'+str(self.spinBox_7.value())+',右：'+str(self.spinBox_8.value()))
            self.textBrowser.append('选择模型：'+self.lineEdit.text())
            self.textBrowser.append('自动驾驶开始......')
            
            #    def run_pre(self,car,uart,error_left = 70,error_right = 50,v_left = 120,v_right = 190,y_bottom = 250,y_top = 300,
            #                        forward_left = 143,forward_right = 143,model_name = 'final_model_30_1.h5',star=1):
            roadPre.run_pre(car,x, v_left = self.spinBox_4.value(),v_right = self.spinBox_3.value(), 
                                forward_left = self.spinBox.value(),forward_right = self.spinBox_2.value(), 
                                model_name = self.lineEdit.text(), star=1)
        else:
             print('串口未打开，请打开串口发送数据。')
             self.textBrowser.append('串口未打开，请打开串口发送数据。')

        
        
def open_uart(ui):
    try:
        uart = MyUart()
        com = uart.gerSerialName()
        x = uart.start(com)
    except:
        print('打开串口失败！！！')
        ui.textBrowser.append('打开串口失败！！！')
    else:
        print('uart is open:', x.is_open)
        print('串口连接:', com)
        print('串口连接成功！')
        ui.textBrowser.append('串口连接成功！')
        
        return x

if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()    
    
    ## 打开串口
    x = open_uart(ui)
    print('小车控制初始化！')
    ui.textBrowser.append('小车控制初始化！')
    car = Car()
    # 开始，或静止
    car.start(x)
    print('小车控制初始化完毕！')
    ui.textBrowser.append('小车控制初始化完毕！')
    print('车道识别初始化！')
    ui.textBrowser.append('车道识别初始化！')
    roadPre = RoadPre()
    print('车道识别初始化完毕！')
    ui.textBrowser.append('车道识别初始化完毕！')

    print('初始化完毕')   
    ui.textBrowser.append('初始化完毕！')
    
    sys.exit(app.exec_())
    
#    try:
#        print('小车控制初始化！')
#        car = Car()
#        # 开始，或静止
#        car.start(x)
#        print('小车控制初始化完毕！')
#        print('车道识别初始化！')
#        roadPre = RoadPre()
#        print('车道识别初始化完毕！')
#                
#    finally:
#            app = QtWidgets.QApplication(sys.argv)
#            ui = MainWindow()
#            ui.show()
#            print('初始化完毕')
#            
#            roadPre.run_pre(car,x)
##            cap = cv2.VideoCapture(0)
#            
##            while True:
##                ret,frame = cap.read()
##                cv2.imshow('frame',frame)
##                if cv2.waitKey(1) & 0xFF == ord('q'):
##                    cap.release()
##                    cv2.destroyAllWindows()
##                    break
#            sys.exit(app.exec_())
            

