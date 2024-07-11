from UVLRM import UVLRM

model = UVLRM(0,0,0.01,100000)
model.trainModel('test.csv')
model.predict(5)
