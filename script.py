from UVLRM import UVLRM

model = UVLRM(0,0,0.01,100000)
model.trainModel('train.csv')
model.predict(5)
