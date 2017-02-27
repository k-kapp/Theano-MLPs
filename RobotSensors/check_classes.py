import pandas as pd

df = pd.read_csv("sensor_readings_2.data", header=None)

classes = list(df[2].values)
cls_set = set(classes)
num_obs = len(classes)

for cls in cls_set:
    print ("For class " + cls + ", proportion is: " + str(classes.count(cls)/num_obs))

