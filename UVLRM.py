import pandas as pd
import numpy as np

# class implementing univariate linear regression
class UVLRM:

    # initialize the values of paramenteres
    def __init__(self, w, b, alpha, numOfIterations):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.numOfIterations = numOfIterations

    # Cost function = J(w,b)
    # Helps to know how well the model is performing
    def computeCost(self):
        cost = 0
        for i in range(0, self.m):
            f_wb = self.w * self.X[i] + self.b
            cost = cost + (f_wb - self.Y[i])**2
        totalCost = cost / (2 * self.m)
    
    # Calculated the gradient to update the actual parameters
    # derivative of cost function w.r.t 'w' and 'b'   
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

    # updates the actual parameters
    def batchGradientDescent(self):
        for i in range(0, self.numOfIterations):
            dj_dw, dj_db = self.computeGradient()
            self.w = self.w - self.alpha * dj_dw 
            self.b = self.b - self.alpha * dj_db

    def Z_Scale_Normalization(self):
        XMean = sum(self.X)/self.m
        YMean = sum(self.Y)/self.m
        XStandardDeviation = 0
        YStandardDeviation = 0
        for i in range(self.m):
            XStandardDeviation += (self.X[i] - XMean)**2
            YStandardDeviation += (self.Y[i] - YMean)**2
        XStandardDeviation = np.sqrt(XStandardDeviation/self.m)
        YStandardDeviation = np.sqrt(YStandardDeviation/self.m)
        for i in range(self.m):
            self.X[i] = (self.X[i] - XMean)/XStandardDeviation
            self.Y[i] = (self.Y[i] - YMean)/YStandardDeviation

    def trainModel(self, fileName):
        trainData = pd.read_csv(fileName)
        self.m = len(trainData)
        self.X = trainData['x'].tolist()
        self.Y = trainData['y'].tolist()
        self.Z_Scale_Normalization()
        self.batchGradientDescent()
    
    def predict(self, x):
        print(f"Prediction = {self.w * x + self.b}")
