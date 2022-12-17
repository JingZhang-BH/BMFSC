import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


IPines = [58.92,65.35,81.84,51.39]
Salinas =[75.05,73.48,95.94,74.53]
PU = [70.77,70.93,85.65,66.03]
x = np.array([5,7,9,11])
print(x)

figsize = 16,8
figure, ax = plt.subplots(figsize=figsize)
# figure = plt.figure()
# ax = plt.subplots(111)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

plt.xticks(x)
# x = x*12 + 380
linew = 4.5
# plt.plot(x, IPines, lw = 3, label="Indian Pines")
# plt.plot(x, Salinas, color = 'orchid', lw = 3, label="Salinas Vally")
# plt.plot(x, PU, color='tab:orange', lw = 3, label="Pavia University")
# plt.plot(x, PC, color='tab:green', lw = 3, label="Pavia Center")
plt.plot(x, IPines, marker = 'o', lw = linew, label="Indian Pines")
plt.plot(x, Salinas, marker = 'o',  lw = linew, label="Salinas Vally")
plt.plot(x, PU,  marker = 'o',  lw = linew, label="Pavia University")
# plt.plot(x, PC, marker = 'o',  lw = linew, label="Pavia Center")
# linestyle=(0, (5, 5)),
# plt.plot(x, tgt_10, lw = 3, label="SNR = 10")
# plt.plot(x, tgt_30, lw = 3, label="SNR = 30")
# plt.plot(x, tgt_50, lw = 3, label="SNR = 50")
# plt.plot(x, tgt, color = 'tab:gray', lw = 2, label="Target spectrum")


font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}


font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}
# plt.legend(prop=font2, loc='lower right', facecolor='none', edgecolor='none', fontsize = 22)

plt.tick_params(labelsize=22)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

bwith = 1.5 #边框宽度设置为2
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)#图框下边
TK.spines['left'].set_linewidth(bwith)#图框左边
TK.spines['top'].set_linewidth(bwith)#图框上边
TK.spines['right'].set_linewidth(bwith)#图框右边
# ax.spines['top'].set_visible(False) #去掉上边框
# ax.spines['bottom'].set_visible(False) #去掉下边框
# ax.spines['left'].set_visible(False) #去掉左边框
# ax.spines['right'].set_visible(False) #去掉右边框

# ax.yaxis.grid(color='grey',
#               linestyle='-',
#               linewidth=2,
#               alpha=0.3)


# plt.xlim([5, 11])
# plt.ylim([0.0, 1.0])
plt.xlabel('Window Size (pixel)', font2)
plt.ylabel('Accuracy (%)', font2)
# plt.legend(prop=font3, loc='SouthEastOutside', facecolor='none', edgecolor='none')
plt.legend(prop=font3, facecolor='none', edgecolor='none', bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
# plt.title('ROC Curve')
# plt.title('Target Spectrum of the San Diego Data Set', family='Times New Roman', fontsize='26',
#               fontweight='normal',
#               loc='center',
#               verticalalignment='top')

figure.subplots_adjust(right=0.785)
plt.show()   # show ROC curve
# figure.savefig('winsize_v3.0.png',)
# plt.savefig('Roc_cup_10.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
# plt.savefig(name_roc, bbox_inches='tight', dpi=300, pad_inches=0.0)

# plt.show()
figure.savefig('winsize_v3.0.png', bbox_inches='tight', dpi=300, pad_inches=0.0)