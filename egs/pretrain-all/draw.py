import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# exp_name = "ende_shrink_AT_baseline"
# exp_name = "ende_shrink_doublecontrastive"
# exp_name = "ende_shrink_baseline_mt_0719"
# exp_name = "ende_shrink_v1_merge_AT_sentence_scale1.5_mt0.5_0728"
# exp_name = "ende_shrink_v1_merge_baseline_mt_0724_2250"
# exp_name = "ende_shrink_v1_merge_large_0812_doubleCL_alpha2.5_mt0.5"


baseline = "ende_shrink_v2_merge_large_0818_baseline_alpha1.5_mt0.5"

exp_name = "ende_shrink_v1_merge_large_0815_AT_sentence_mixup0105_mt0102_id0_alpha1.5_mt0.5"
exp_name = "ende_shrink_v2_merge_large_0818_doubleCL_alpha1.5_mt0.5"


level = 'sentence'

pic_dir = '/mnt/zhangyh/fairseq-AT/egs/pretrain-all/pic'
temp_dir = '/mnt/zhangyh/fairseq-AT/egs/pretrain-all/pic/temp'

spe_temp_file = temp_dir + '/' + exp_name + '_' + level + '_speech.pt'
txt_temp_file = temp_dir + '/' + exp_name + '_' + level + '_text.pt'

pca_file = temp_dir + '/' + baseline + '_pca_model.pkl'
std_file = temp_dir + '/' + baseline + '_std_model.pkl'
# minmax_file = temp_dir + '/' + 'ende_shrink_AT_baseline' + '_minmax_model.pkl'

pic_file_input = pic_dir + '/' + exp_name + '_' + level + '_pca_input.png'
pic_file_output = pic_dir + '/' + exp_name + '_' + level + '_pca_output.png'

speech = torch.load(spe_temp_file)  # [size, dim]
text = torch.load(txt_temp_file)    # [size, dim]
print("Length:\t", speech['input'].shape,  speech['output'].shape, text['input'].shape,  text['output'].shape)

assert speech['input'].shape[0] == 2587
assert speech['output'].shape[0] == 2587
assert text['input'].shape[0] == 2587
assert text['output'].shape[0] == 2587

speech_input = speech['input'].detach().numpy()
speech_output = speech['output'].detach().numpy()
text_input = text['input'].detach().numpy()
text_output = text['output'].detach().numpy()


STD = False
# PCA
std = StandardScaler()
pca = PCA(n_components=2)

if exp_name == baseline and False:
    std.fit(speech_output)
    std.fit(text_output)
    joblib.dump(std, std_file)

    pca.fit(std.transform(speech_output))
    pca.fit(std.transform(text_output))
    joblib.dump(pca, pca_file)
    
else:
    std = joblib.load(std_file)
    pca = joblib.load(pca_file)
    

speech_input = pca.transform(std.transform(speech_input))
speech_output = pca.transform(std.transform(speech_output))

text_input = pca.transform(std.transform(text_input))
text_output = pca.transform(std.transform(text_output))

print(speech_input.shape, speech_output.shape, text_input.shape, text_output.shape)

speech_id = np.ones((speech_output.shape[0], 1))
text_id = np.zeros((text_output.shape[0], 1))
# print(speech_id.shape, text_id.shape)

speech_input = np.concatenate((speech_input, speech_id), axis=1)
speech_output = np.concatenate((speech_output, speech_id), axis=1)
text_input = np.concatenate((text_input, text_id), axis=1)
text_output = np.concatenate((text_output, text_id), axis=1)
print(speech_input.shape, speech_output.shape, text_input.shape, text_output.shape)

speech_input_data = pd.DataFrame(speech_input, columns=["X", "Y", "Key"])
speech_output_data = pd.DataFrame(speech_output, columns=["X", "Y", "Key"])
text_input_data = pd.DataFrame(text_input, columns=["X", "Y", "Key"])
text_output_data = pd.DataFrame(text_output, columns=["X", "Y", "Key"])

input_data = pd.DataFrame(np.vstack((speech_input, text_input)), columns=["X", "Y", "Key"])
output_data = pd.DataFrame(np.vstack((speech_output, text_output)), columns=["X", "Y", "Key"])

mpl.rc("figure", figsize=(6, 6))
with sns.axes_style('white'):
    # sns.jointplot(text_data, x='X', y='Y', kind="hex", xlim=(-17,22), ylim=(-15,22),
    #              )
    # sns.jointplot(speech_data, x='X', y='Y', kind="kde", xlim=(-17,22), ylim=(-15,22), 
    #               fill=True, bw_method="silverman", gridsize=50)
    sns.jointplot(input_data, x='X', y='Y', hue="Key", kind="kde", xlim=(-75,75), ylim=(-25,15), 
                  fill=False, bw_method="silverman", gridsize=50)

plt.savefig(pic_file_input)
print("Draw Pic:\t", pic_file_input)

mpl.rc("figure", figsize=(6, 6))
with sns.axes_style('white'):
    # sns.jointplot(text_data, x='X', y='Y', kind="hex", xlim=(-17,22), ylim=(-15,22),
    #              )
    # sns.jointplot(speech_data, x='X', y='Y', kind="kde", xlim=(-17,22), ylim=(-15,22), 
    #               fill=True, bw_method="silverman", gridsize=50)
    sns.jointplot(output_data, x='X', y='Y', hue="Key", kind="kde", xlim=(-75,75), ylim=(-25,15), 
                  fill=False, bw_method="silverman", gridsize=50)

plt.savefig(pic_file_output)
print("Draw Pic:\t", pic_file_output)



"""
# x_s = speech[:,0]
# y_s = speech[:,1]
x_t = text[:,0]
y_t = text[:,1]

# plt.scatter(x_s, y_s, color='red', label='Speech')
# plt.scatter(x_t, y_t, color='blue', label='Text')

plt.hist2d(x_t, y_t,
           bins=100, 
           norm = 'log',
           cmap='plasma')
plt.colorbar()

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')

plt.savefig(pic_file)
print("Draw Pic:\t", pic_file)
# plt.show()
"""

"""speech = torch.from_numpy(speech[:,:2])
text = torch.from_numpy(text[:,:2])

mean_speech = speech.mean(dim=0)
mean_text = text.mean(dim=0)
var_speech = speech.var(dim=0)
var_text = text.var(dim=0)

distance = torch.norm(mean_speech - mean_text)
print("Mean Distance:\t", distance)
print("Var Speech&Text:\t", var_speech, var_text)"""

"""
# KL DIV
p = speech
q = text
# kl_div = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)
kl_div = F.kl_div(p.log(), q, reduction='sum')
print("KL Div:\t", kl_div)
"""
