
# coding: utf-8

# In[2]:


import serial.tools.list_ports

class MyUart():
    
    def __init__(self,):
        self.slist = None
        self.serialName = None
        self.slist_0 = None
        self.ser = None
        self.serialFd = None
        print('Uart init OK!')
        
    def gerSerialName(self,):
        self.slist = list(serial.tools.list_ports.comports())
        if len(self.slist) <= 0:
            print('没有发现端口')
        else:
            self.slist_0= list(self.slist[0])
            self.serialName = self.slist_0[0]
            
            return self.serialName
			
    def start(self,com):
        
        ser = serial.Serial()  
        ser.baudrate = 9600 #设置波特率 
        ser.port = com #端口是COM3  
        print(ser)  
        ser.open()#打开串口  
        print(ser.is_open)#检验串口是否打开     
        return ser

            

