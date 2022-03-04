#


#import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#        0                                                                                           


f1 = [0.7259672837692231, 0.7385002233453621, 0.7840485074626866, 0.8037900564249973, 0.6405228758169934, 0.9407433033507957, 0.9040241448692152, 0.7138727191811304, 0.9617039964866052, 0.7906153630501822, 0.7316166531494999, 0.7346059113300492, 0.9334031048316762, 0.8244443316756318, 0.9353213507625272, 0.8041429002872313, 0.7795621650812113, 0.7349505840071877, 0.8438229897156513, 0.9410868994486743]

f1_repr = [0.44, 0.45, 0.60, 0.63, 0.44, 0.86, 0.85, 0.48, 0.54, 0.59, 0.55, 0.55, 0.83, 0.68, 0.74, 0.49, 0.65, 0.41, 0.79, 0.78]

f1_repr_para = [0.48761905, 0.57283289, 0.70704443, 0.75632707, 0.44661654,
       0.86238416, 0.85635234, 0.62583815, 0.71427775, 0.62957579,
       0.62575758, 0.65605394, 0.84863222, 0.68287689, 0.79228382,
       0.60913031, 0.64636454, 0.59292865, 0.78133982, 0.79955864]
               
    
f1_repr_cnn = [0.63094194, 0.64823727, 0.57947321, 0.76699446, 0.52312988,
       0.87953657, 0.86198558, 0.61391155, 0.84007957, 0.6900991 ,
       0.65206629, 0.68241437, 0.9008405 , 0.72596758, 0.87422665,
       0.70833052, 0.68210967, 0.65585319, 0.80183817, 0.870214  ] 

f1_repr_cnn_para = [0.61593535, 0.63853716, 0.72135839, 0.77720498, 0.52545985,
       0.86358209, 0.86004433, 0.61198564, 0.78006092, 0.71548001,
       0.65778515, 0.65001226, 0.86517499, 0.66970068, 0.84197707,
       0.63686306, 0.66480012, 0.69675252, 0.80079129, 0.82818255]

f1_repr_cnn_para_1 = [0.44924569, 0.61289079, 0.61289858, 0.72757447, 0.5113994 ,
       0.86975562, 0.85884734, 0.65771958, 0.7296627 , 0.65369664,
       0.62487732, 0.67384425, 0.85674432, 0.68269769, 0.78328967,
       0.63821799, 0.65196669, 0.6515118 , 0.77079956, 0.78282623]

f1_repr_uni = [0.4986785 , 0.57535035, 0.71507692, 0.74248964, 0.44006999,
       0.88491704, 0.86198558, 0.58009051, 0.73344324, 0.66221784,
       0.64120543, 0.65614485, 0.86001012, 0.70810585, 0.79499606,
       0.63479198, 0.67416119, 0.60137097, 0.7736633 , 0.84087199]

f1_repr_nor = [0.45599721, 0.58054867, 0.70910058, 0.72856295, 0.46315969,
       0.86359655, 0.85635234, 0.61162929, 0.70677144, 0.6357537 ,
       0.62861009, 0.65793651, 0.85015823, 0.68231668, 0.80374604,
       0.61543517, 0.66636916, 0.65540218, 0.78228146, 0.81852617]

ins = {"accordion": 0, "banjo": 1, "bass": 2, "cello": 3, "clarinet": 4, "cymbals": 5, "drums": 6, "flute": 7, "guitar": 8, "mallet_percussion": 9, "mandolin": 10, "organ": 11, "piano": 12, "saxophone": 13, "synthesizer": 14, "trombone": 15, "trumpet": 16, "ukulele": 17, "violin": 18, "voice": 19}
ins_name = ["accordion", "banjo", "bass", "cello", "clarinet", "cymbals", "drums", "flute", "guitar", "mallet_percussion", "mandolin", "organ", "piano", "saxophone", "synthesizer", "trombone", "trumpet", "ukulele", "violin", "voice"]

index = np.arange(20)
bar_width = 0.3
plt.bar(index, f1_repr, bar_width)
plt.bar(index+bar_width, f1_repr_para, bar_width)
plt.bar(index+bar_width+bar_width, f1_repr_cnn, bar_width)

#plt.grid(True)
plt.title('F1 score comparison over the instruments')
plt.legend(['Baseline','Adversarial','CNN'])
plt.xticks((index + bar_width /2 ), ins_name)
#plt.xticklabels(ins_name)
#plt.xlabel('Instrument')
plt.xticks(rotation=90)
plt.ylim((0.3, 1)) 
plt.ylabel('F1 score')
plt.tight_layout()
plt.savefig('./Ins_f1_parameter_all_cmp.pdf')
plt.show()

print('avg F1: ', np.mean(f1_repr_cnn_para_1))

f1_bs = [0.6942613016299292, 0.7308396178984414, 0.7856481481481481, 0.806, 0.6116432345637521, 0.9313459346085773, 0.9029476843256441, 0.7098458406050029, 0.9614493624997789, 0.7781805446108045, 0.737407058931522, 0.6746002676095293, 0.944985173556602, 0.8342018289097508, 0.9417892156862744, 0.8058734411243591, 0.782384856045305, 0.7497422377677139, 0.839793500338524, 0.9436831802439392]
'''
index = np.arange(20)
bar_width = 0.4
plt.bar(index, f1, bar_width)
plt.bar(index+bar_width, f1_bs, bar_width)

#plt.grid(True)
plt.title('Baseline/FixMatch F1 score over the instruments')
plt.legend(['FixMatch','Baseline'])
plt.xticks((index + bar_width / 2), ins_name)
#plt.xticklabels(ins_name)
#plt.xlabel('Instrument')
plt.xticks(rotation=90)
plt.ylim((0.4, 1)) 
plt.ylabel('F1 score')
plt.tight_layout()
plt.savefig('./plot/Ins_f1_bs.pdf')
plt.show()
'''
'''
pos_label = [ 489.,  732.,  549.,  824.,  533., 1111., 1106.,  647., 1138., 733.,  845.,  603., 1170., 1135., 1091.,  863., 1146.,  738., 1173.,  988.]

neg_label = [1582., 1486., 1339., 1125., 1852.,  624.,  641., 1437.,  512., 1069., 1619., 1287.,  550., 1230.,  511., 1897., 1770., 1687., 860.,  576.]

sum_pos = sum(pos_label)
pos_label_norm = [num/sum_pos for num in pos_label]

#f1_dif = f1 - f1_bs
f1_dif = []
zip_object = zip(f1, f1_bs)
for list1_i, list2_i in zip_object:
    f1_dif.append(list1_i-list2_i)

index = np.arange(20)
bar_width = 0.4
plt.bar(index, f1_dif, bar_width)
plt.bar(index+bar_width, pos_label_norm, bar_width)
#plt.grid(True)
plt.title('FixMatch F1 improvement vs pos label number')
plt.legend(['F1 difference','Pos label'])
plt.xticks((index + bar_width / 2), ins_name)
#plt.xticklabels(ins_name)
#plt.xlabel('Instrument')
plt.xticks(rotation=90)
#plt.ylim((0, 0.07)) 
plt.ylabel('F1 score improvement')
plt.tight_layout()
plt.savefig('./plot/Ins_f1_pos.pdf')
plt.show()
'''



