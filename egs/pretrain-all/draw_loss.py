import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
from scipy.ndimage import gaussian_filter1d


# exp_name = "ende_v3_merge_wmt_0902_shrink_soft_noCL_AT_sentence_mixup_changeid_scale3.5_alpha1.5_mt0.5"
exp_name = "ende_v3_merge_wmt_0901_shrink_soft_noCL_AT_sentence_scale3.5_alpha1.5_mt0.5"

task_loss_name = "run-train_inner-tag-task_loss.csv"
task_gen_loss_name = "run-train_inner-tag-task_loss_gen.csv"

TL_file = "/mnt/zhangyh/fairseq-AT/egs/pretrain-all/task_loss/" + exp_name + '/' + task_loss_name
TGL_file = "/mnt/zhangyh/fairseq-AT/egs/pretrain-all/task_loss/" + exp_name + '/' + task_gen_loss_name

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        if row[1] != "Step":
            y.append(float(row[2]))
            x.append(float(row[1]))
    print(len(x), len(y))
    return x, y

show_range= [200, 400]
TL_steps, TL_val  = readcsv(TL_file)
TGL_steps, TGL_val = readcsv(TGL_file)

TL_steps = TL_steps[show_range[0]:show_range[1]]
TL_val =  [(i / 3.5) for i in TL_val[show_range[0]:show_range[1]]]
TGL_steps = TGL_steps[show_range[0]:show_range[1]]
TGL_val = [(i / 3.5) for i in TGL_val[show_range[0]:show_range[1]]]


TL_val = gaussian_filter1d(TL_val, sigma=1)
TGL_val = gaussian_filter1d(TGL_val, sigma=1)


plt.figure()
plt.subplots(figsize=(6,3))
l1, = plt.plot(TL_steps, TL_val, color='red', linewidth=1.5)
l2, = plt.plot(TGL_steps, TGL_val, color='blue', linewidth=1.5)
plt.legend(handles=[l1,l2],labels=['Dis Loss','Gen Loss'], fontsize=15)
plt.ylim(0, 4)
plt.xlim(show_range[0]*100, show_range[1]*100)
plt.xticks(fontsize=15, ticks = range(show_range[0]*100, show_range[1]*100 + 1, 5000))
plt.yticks(fontsize=15)
plt.xlabel("Steps", fontsize=15)
plt.ylabel("Loss", fontsize=15)

plt.savefig("/mnt/zhangyh/fairseq-AT/egs/pretrain-all/task_loss/" + exp_name + '.png')


print_List1 = []
print_List2 = []
for i in range(0, len(TL_steps), 5):
    print_List1.append((TL_steps[i], TL_val[i]))
    print_List2.append((TGL_steps[i], TGL_val[i]))

for ls in print_List1:
    print("({}, {:.4f})".format(ls[0]/1000, ls[1]), end=' ')
print('\n\n')
for ls in print_List2:
    print("({}, {:.4f})".format(ls[0]/1000, ls[1]), end=' ')


