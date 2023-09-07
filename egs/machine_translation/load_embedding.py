import torch
import fairseq
import numpy as np

arg_overrides={}
state = fairseq.checkpoint_utils.load_checkpoint_to_cpu("checkpoints/wmt-en2fr/enfr-baseline/last5.ensemble.pt", arg_overrides)
dicts=open("/mnt/zhangyh/fairseq-AT/egs/machine_translation/data-bin/wmtmerge2-enfr/dict.fr.txt","r").read().strip().split("\n")
output=open("pretrain_embeddings_wmt_enfr_baseline","w")
dicts=["<s>","<pad>","</s>","<unk>"]+dicts
output.write(str(len(dicts))+" 512\n")
for key in list(state['model'].keys()):
    if key == "decoder.embed_tokens.weight":
        embedding=state["model"][key].data
        for index in range(len(dicts)):
            output.write(dicts[index].split(" ")[0]+" ")
            feature=embedding[index:index+1,].numpy()
            np.savetxt(output,feature)
        break
