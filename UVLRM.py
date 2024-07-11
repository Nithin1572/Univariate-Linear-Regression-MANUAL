import pandas as pd
import numpy as np

class UVLRM:
    def __init__(self, w, b, alpha, numOfIterations):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.numOfIterations = numOfIterations

    def computeCost(self):
        cost = 0
        for i in range(0, self.m):
            f_wb = self.w * self.X[i] + self.b
            cost = cost + (f_wb - self.Y[i])**2
        totalCost = cost / (2 * self.m)
    
    def computeGradient(self):
        dj_dw = 0
        dj_db = 0
        for i in range(self.m):
            f_wb = self.w * self.X[i] + self.b
            dj_dw = dj_dw + (f_wb - self.Y[i])* self.X[i]
            dj_db = dj_db + (f_wb - self.Y[i])
        dj_dw = dj_dw / self.m
        dj_db = dj_db / self.m
        return dj_dw, dj_db

    def gradientDescent(self):
        for i in range(0, self.numOfIterations):
            dj_dw, dj_db = self.computeGradient()
            self.w = self.w - self.alpha * dj_dw 
            self.b = self.b - self.alpha * dj_db


    def trainModel(self, fileName):
        trainData = pd.read_csv(fileName)
        self.m = len(trainData)
        self.X = trainData['x'].tolist()
        self.Y = trainData['y'].tolist()
        maxElementX = max(self.X)
        minElementX = min(self.X)
        maxElementY = max(self.Y)
        minElementY = min(self.Y)
        for i in range(self.m):
            self.X[i] = (self.X[i] - minElementX)/(maxElementX - minElementX)
            self.Y[i] = (self.Y[i] - minElementY)/(maxElementY - minElementY)
        self.gradientDescent()
    
    def predict(self, x):
        print(f"Prediction = {self.w * x + self.b}")
