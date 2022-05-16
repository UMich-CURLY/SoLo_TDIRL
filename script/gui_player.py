import tkinter as tk
from laser2density import Laser2density
from matplotlib.figure import Figure
import numpy as np
from geometry_msgs.msg import PoseStamped
import rospy
from tkinter import * 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import matplotlib.pyplot as plt

class GuiPlayer(tk.Frame):
   def __init__(self, parent, gridsize=(250,250),resolution=0.001):
       tk.Frame.__init__(self, parent)
       self.parent = parent
       self.initialize_user_interface()
       self.laser = Laser2density(gridsize=gridsize, resolution=resolution)

   def initialize_user_interface(self):
       self.parent.geometry("1000x800")
       self.parent.title("Fetch IRL Interactive GUI")
    #    self.entry=tk.Entry(self.parent)
    #    self.entry.pack()
    #    self.button=tk.Button(self.parent,text="Enter", command=self.PassCheck)
    #    self.button.pack()
    #    self.label=tk.Label(self.parent,text="Please a password")
    #    self.label.pack()

   def plot(self,map_logs):
       fig = Figure(figsize = (5, 5), dpi = 100)
       plot1 = fig.add_subplot(111)
       plot1.imshow(1.0 - 1./(1.+np.exp(map_logs)), 'Greys')
       canvas = FigureCanvasTkAgg(fig, master = self.parent)  
       canvas.draw() 
       canvas.get_tk_widget().pack()
       toolbar = NavigationToolbar2Tk(canvas, self.parent)
       toolbar.update()
       canvas.get_tk_widget().pack()



if __name__ == '__main__':
   rospy.init_node("GUI",anonymous=False)
   data = PoseStamped()
   data.pose.position.x = 6
   data.pose.position.y = -6
   data.header.frame_id = "/map"
   root = tk.Tk()
   gui = GuiPlayer(root)
   gui.initialize_user_interface()
   gridsize=(250,250)
   while not rospy.is_shutdown():
    map_logs = np.reshape(gui.laser.result,(gridsize[1], gridsize[0]))
    gui.plot(map_logs)
    gui.mainloop()
    rospy.spin()
