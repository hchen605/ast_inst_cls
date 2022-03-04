
#import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#        0                                                                                           


f1 = [62.03, 68.46, 69.39, 72.94, 0]
f1_2 = [0,0,0,0,81.3]
bar_width = 0.5
method = ['Baseline','Adversarial','Noise','CNN','FixMatch']
plt.bar(method,f1,bar_width)
plt.bar(method,f1_2,bar_width)
#plt.legend(['Baseline','Const','CNN'])
plt.title('OpenMic Instrument Classification')
plt.ylim((50, 85)) 
plt.xlabel('Reprogramming method')
plt.ylabel('F1 score %')
plt.savefig('./f1_trend.pdf')

