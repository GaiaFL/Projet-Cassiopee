import matplotlib.pyplot as plt
import numpy as np

# creating the dataset
data = {'1rst - all floors':5.56, '1rst - one floor':44.4, '2nd':36.67,
        '3rd':28.57} #experiments done
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Experiment")
plt.ylabel("Best prediction percentage (%)")
plt.title("Percentage of success for each experiment")
plt.show()