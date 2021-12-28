# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import mxnet as mx
from mxnet import nd
from sklearn import preprocessing
import numpy as np
from collections import Counter
import os
import glob
import random
import argparse 
import fileinput
import mxnet.gluon.data as gdata
from ui import Ui_MainWindow

def scoreToClass(up,down,y_predict7_score_cnn):
	y_predict7_cnn=[]
	for i in range(len(y_predict7_score_cnn)):
		if y_predict7_score_cnn[i]>=up:
			y_predict7_cnn.append(2)
		elif y_predict7_score_cnn[i]<=down:
			y_predict7_cnn.append(0)
		elif (y_predict7_score_cnn[i]>down) and (y_predict7_score_cnn[i]<up):
			y_predict7_cnn.append(1)
	return y_predict7_cnn

def create__file(file_path,msg):
	f=open(file_path,"a")
	f.write(msg)
	f.close

def Forward(peptide_list, up, down):

	ctx = mx.cpu()
	
	net = mx.gluon.nn.HybridSequential()
	net.add(mx.gluon.nn.Conv2D(channels=64, kernel_size=(2,7), activation='relu'),
			mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
			mx.gluon.nn.Conv2D(channels=128, kernel_size=(2,3), activation='relu'),
			mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
			mx.gluon.nn.Conv2D(channels=256, kernel_size=(2,3),padding=1,activation='relu'),
			mx.gluon.nn.MaxPool2D(pool_size=2, strides=1),
			mx.gluon.nn.Dense(512, activation='relu'),
			mx.gluon.nn.Dropout(0.3),
			mx.gluon.nn.Dense(256, activation='relu'),
			mx.gluon.nn.Dropout(0.3),
			mx.gluon.nn.Dense(1,activation='sigmoid'))
	
	net.load_parameters("cnn_shuffle.mxnet", ctx=ctx)
	net.hybridize()
	X = []
	Pep_name=[]
	for pep_name in peptide_list:
		pep_data = peptide_list[pep_name]
		x=[]
		for i in range(1,7):
			line = pep_data[i]
			if len(line) < 85:
				padding = 85 - len(line)
				half_padding = padding // 2
				line = [0] * half_padding + line + [0] * (padding - half_padding)
			elif len(line) >= 85:
				line = line[0: 85]
			x.append(line)
		x = preprocessing.minmax_scale(x, axis=1)
		X.append(x)
		Pep_name.append(pep_name)
	X_scaled_nd = nd.array(X).reshape(len(X), 1, 6, 85)
	data_iter = gdata.DataLoader(X_scaled_nd, batch_size=256, shuffle=False)
	Y_score = []
	y_score = []
	for data in data_iter:
		data = data.as_in_context(ctx)
		Y_score.extend(net(data).asnumpy().tolist())
	for i in range(len(Y_score)):
		y_score.append(Y_score[i][0])
	y = scoreToClass(up, down, y_score)
	COUTER = Counter(y)
	return Pep_name, y_score, y, COUTER

import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QAbstractItemView

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Figure_Canvas(FigureCanvas):   # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplot                                          lib的关键
	def __init__(self, parent=None, width=7.5, height=4.7, dpi=100):
		fig = Figure(figsize=(width, height), dpi=100)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
		FigureCanvas.__init__(self, fig) # 初始化父类
		self.setParent(parent)
		self.axes = fig.add_subplot(111) # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
		self.Fig = fig
		self.pep_seq = ""
		
	def showImage(self, x, seq):
		self.axes.clear()
		self.axes.set_title(seq)
		
		self.pep_seq = seq
		
		for i in range(1, len(x)):
			self.axes.plot(x[0], x[i])
		
	def saveImage(self, save_dir):
		self.Fig.savefig(save_dir + self.pep_seq + ".png", format='png', transparent=True, dpi=300, pad_inches = 0)

class MyMainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super(MyMainWindow, self).__init__(parent)

		self.setupUi(self)

		self.pushButton.clicked.connect(self.openMzXML)
		self.pushButton_2.clicked.connect(self.add)
		self.pushButton_3.clicked.connect(self.remove)
		self.pushButton_4.clicked.connect(self.openAlignedTSV)
		self.pushButton_5.clicked.connect(self.openLibOSTSV)
		self.pushButton_6.clicked.connect(self.openWinOSTSV)
		self.pushButton_8.clicked.connect(self.extractIC)
		
		self.horizontalSlider.valueChanged.connect(self.valChange)
		self.horizontalSlider_2.valueChanged.connect(self.valChange_2)
		self.lineEdit_2.returnPressed.connect(self.setSlider)
		self.lineEdit_3.returnPressed.connect(self.setSlider_2)
		self.OutputBrowse.clicked.connect(self.openFile_save)
		
		self.pushButton_7.clicked.connect(self.Plot)
		
		self.Save_2.clicked.connect(self.SaveFigure)
		
		self.start.clicked.connect(self.Start)
		self.Save.clicked.connect(self.save)
		
	def openMzXML(self):
		get_filenames_path, ok = QFileDialog.getOpenFileNames(self,
								"Select mzXML files")
		if ok:
			self.lineEdit.setText(str('?'.join(get_filenames_path)))

	def openAlignedTSV(self):
		get_filenames_path, ok = QFileDialog.getOpenFileName(self,
								"Select aligned.tsv")
		if ok:
			self.lineEdit_4.setText(str(''.join(get_filenames_path)))
	
	def openLibOSTSV(self):
		get_filenames_path, ok = QFileDialog.getOpenFileName(self,
								"Select lib.os.tsv")
		if ok:
			self.lineEdit_5.setText(str(''.join(get_filenames_path)))

	def openWinOSTSV(self):
		get_filenames_path, ok = QFileDialog.getOpenFileName(self,
								"Select win.os.tsv")
		if ok:
			self.lineEdit_6.setText(str(''.join(get_filenames_path)))


	def valChange(self):
		self.lineEdit_2.setText(str(round(self.horizontalSlider.value()*1e-5, 5)))
		#print(self.horizontalSlider.value())
	def valChange_2(self):
		self.lineEdit_3.setText(str(round(self.horizontalSlider_2.value()*1e-5, 5)))

	def add(self):
		a = self.lineEdit.text().split('?')
		for i in a:
			self.listWidget.addItem(i)
			self.lineEdit.clear()
	def remove(self):
		rows = self.listWidget.selectedItems()
		#print(rows)
		for row in range(len(rows)):
			item=self.listWidget.currentRow()
			self.listWidget.takeItem(item)

	def openFile_save(self):
		get_directory_path = QFileDialog.getExistingDirectory(self,
								"Select files")
		self.OutputDirectoryLineEdit.setText(str(get_directory_path))
	
	def setSlider(self):
		value=int(float(self.lineEdit_2.text())*1e5)
		self.horizontalSlider.setValue(value)
		#print(value)
	def setSlider_2(self):
		value=int(float(self.lineEdit_3.text())*1e5)
		self.horizontalSlider_2.setValue(value)
		
	def extractIC(self):
		alignedtsv = self.lineEdit_4.text()
		libostsv = self.lineEdit_5.text()
		winostsv = self.lineEdit_6.text()
		length = self.spinBox.value()
		
		out_path = self.OutputDirectoryLineEdit.text()
		
		self.fig_path = out_path + '/figures/'
		if os.path.exists(self.fig_path) == False:
			os.makedirs(self.fig_path)
		
		xic_path = out_path + "/xic/"
		
		if os.path.exists(xic_path) == False:
			os.makedirs(xic_path)
		
		os.popen("javac -d . PrepareDraw.java")
		os.system("java PrepareDraw " + xic_path + " " + alignedtsv+" " + libostsv + " " + str(length))
		
		xicFiles = glob.glob(xic_path+"*.xic")
		xicFiles = [xic.replace('\\', '/') for xic in xicFiles]
		
		mzxml_list = []
		count = self.listWidget.count()
		for row in range(count):
			mzxml_list.append(self.listWidget.item(row).text())
		
		for xic in xicFiles:
			
			for line in fileinput.input(files=xic, inplace=True):
				
				if fileinput.isfirstline():
					
					while line[-1] == '\n' or line[-1] == '\r':
						line = line[:-1]
					
					mzxml_file = line.split('\\')[-1]
					
					for Path in mzxml_list:
						if mzxml_file in Path:
							print(Path)
				else:
					print(line[:-1])
			
			fileinput.close()
		
		for xic in xicFiles:
			os.system("java PrepareDraw ./xic/ " + " " + xic + " " + winostsv)
		
		txt_out_list = glob.glob(xic_path + '*.txt.out')
		txt_inf_list = glob.glob(xic_path + '*.inf.out')
		
		self.peptide_list = {}
		self.peptide_list_mz = {}
		
		for txt_inf in txt_inf_list:
			mz_data=[]
			with open(txt_inf, 'r') as File:
				for line in File:
					if "#BEG" in line:
						seq = line.split('\t')[1]

					if line[0].isdigit():
						if len(line[:-2].split('\t')) == 6:
							mz = [float(x) for x in line[:-2].split('\t')]
							mz_data.append(mz)
					
					if "#END" in line:
						self.peptide_list_mz[seq] = mz_data
						mz_data = []
						
		for txt_out in txt_out_list:
			
			xic_data = []
			
			with open(txt_out, 'r') as File:
				
				for line in File:
					if "#BEG" in line:
						seq = line.split('\t')[1]
						
					if line[0].isdigit():
						
						spl_line = line[:-2].split('\t')
						spl_line = [float(x) for x in spl_line]
						xic_data.append(spl_line)
					
					if "#END" in line:
						self.peptide_list[seq] = xic_data
						xic_data = []
			
		for seq in self.peptide_list:
			self.listWidget_2.addItem(seq)
		
	def Plot(self):
		
		row = self.listWidget_2.selectedItems()

		if len(row) > 0:
			
			seq = row[0].text()
			data = self.peptide_list[seq]
			
			self.dr = Figure_Canvas()
			self.dr.showImage(data, seq)  # 画图
			graphicscene = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
			graphicscene.addWidget(self.dr)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
			self.graphicsView.setScene(graphicscene) # 第五步，把QGraphicsScene放入QGraphicsView
			self.graphicsView.show()
	
	def SaveFigure(self):
		self.dr.saveImage(self.fig_path)
		
	def Start(self):
		"""
		nameList = ''
		for i in range(self.listWidget.count()):
			item = self.listWidget.item(i)
			name = item.text()
			nameList += name + ','
		"""
		up = float(self.lineEdit_2.text())
		down = float(self.lineEdit_3.text())
		
		self.Pep_name, self.y_score, self.y, self.COUTER = Forward(self.peptide_list, up, down)
		
		negNum = self.COUTER[0]
		fuzNum = self.COUTER[1]
		posNum = self.COUTER[2]
		All = posNum + negNum + fuzNum
		posRatio = posNum / float(All)
		negRatio = negNum / float(All)
		fuzRatio = fuzNum / float(All)
		self.PositiveNum.append(str(posNum))
		self.NegativeNum.append(str(negNum))
		self.FuzzyNum.append(str(fuzNum))
		self.AllNum.append(str(All))
		self.PositiveRatio.append(str(round(posRatio * 100, 2))+str('%'))
		self.FuzzyRatio.append(str(round(fuzRatio * 100, 2))+str('%'))
		self.NegativeRatio.append(str(round(negRatio * 100, 2))+str('%'))
		self.AllRatio.append('100%')
		"""
		
		
		with open('score.txt', 'r') as File:
			line = File.readline().split('\t')
			if len(line[-1]) == 0:
				line = line[:-1]
		
		negNum = float(line[0])
		fuzNum = float(line[1])
		posNum = float(line[2])

		all = posNum + negNum + fuzNum
		posRatio = posNum / float(all)
		negRatio = negNum / float(all)
		fuzRatio = fuzNum / float(all)

		self.PositiveNum.append(str(posNum))
		self.NegativeNum.append(str(negNum))
		self.FuzzyNum.append(str(fuzNum))
		self.AllNum.append(str(all))
		self.PositiveRatio.append(str(round(posRatio * 100, 1))+str('%'))
		self.FuzzyRatio.append(str(round(fuzRatio * 100, 1))+str('%'))
		self.NegativeRatio.append(str(round(negRatio * 100, 1))+str('%'))
		self.AllRatio.append('100%')
        """
	def save(self):
		catogory = []
		for i in range(len(self.y)):
			if self.y[i]==2:
				catogory.append('P')
			if self.y[i]==1:
				catogory.append('F')
			if self.y[i]==0:
				catogory.append('N')
		file_path = self.OutputDirectoryLineEdit.text()
		os.makedirs(file_path + '/PredictedData')
		os.makedirs(file_path + '/PredictedData' + '/PositivePeptide')
		os.makedirs(file_path + '/PredictedData' + '/NegativePeptide')
		os.makedirs(file_path + '/PredictedData' + '/FuzzyPeptide')
		
		df_peptide = pd.DataFrame()
		df_peptide['peptide name'] = self.Pep_name
		df_peptide['probability'] = self.y_score
		df_peptide['class'] = catogory
		df_peptide.to_excel(file_path + '/PredictedData/predictedData.xlsx', index=False)
		
		for i in range(len(self.Pep_name)):
			df_peptide_mz = pd.DataFrame()
			mz_list = self.peptide_list_mz[self.Pep_name[i]]
			df_peptide_mz['retention time'] = self.peptide_list[self.Pep_name[i]][0]
			df_peptide_mz[str(mz_list[0][0])] = self.peptide_list[self.Pep_name[i]][1]
			df_peptide_mz[str(mz_list[0][1])] = self.peptide_list[self.Pep_name[i]][2]
			df_peptide_mz[str(mz_list[0][2])] = self.peptide_list[self.Pep_name[i]][3]
			df_peptide_mz[str(mz_list[0][3])] = self.peptide_list[self.Pep_name[i]][4]
			df_peptide_mz[str(mz_list[0][4])] = self.peptide_list[self.Pep_name[i]][5]
			df_peptide_mz[str(mz_list[0][5])] = self.peptide_list[self.Pep_name[i]][6]
			#print(str(mz_list[0][5]),self.peptide_list[self.Pep_name[i]][6])
			if self.y[i]==0:
				df_peptide_mz.to_excel(file_path + '/PredictedData/NegativePeptide/'+self.Pep_name[i]+'.xlsx', index=False)
			if self.y[i]==1:
				df_peptide_mz.to_excel(file_path + '/PredictedData/FuzzyPeptide/'+self.Pep_name[i]+'.xlsx', index=False)
			if self.y[i]==2:
				df_peptide_mz.to_excel(file_path + '/PredictedData/PositivePeptide/'+self.Pep_name[i]+'.xlsx', index=False)
		
		"""
		with open('score.csv', 'r') as File:
			
			for i, line in enumerate(File):
				
				Line = line.split(',')
				
				fileName = Line[0]
				y = int(Line[1])
				
				command = 'copy ' + fileName + ' ' + file_path + '/PredictedData'
				
				if y == 0:
					command = command + '/NegativeSample/' + fileName.split('/')[-1]
					
				if y == 1:
					command = command + '/FuzzySample/' + fileName.split('/')[-1]
					
				if y == 2:
					command = command + '/PositiveSample/' + fileName.split('/')[-1]
				
				os.system(command)
		
		os.system('copy score.csv ' + file_path + '/PredictedData/predicted_data.csv')
		"""
		
if __name__ == "__main__":
	# 添加if判断语句可以解决spyder内核重启的问题
	if not QtWidgets.QApplication.instance():
		app = QtWidgets.QApplication(sys.argv)
	else:
		app = QtWidgets.QApplication.instance()  
	myWin = MyMainWindow()
	myWin.show()
	sys.exit(app.exec_())
	