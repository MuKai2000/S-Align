import sys
import copy
fo1 = open("train_st.tsv","r")
fo2 = open("train_mt.tsv","w")
fo3 = open("train_asr.tsv","w")
prefix=fo1.readline()
fo2.write(prefix)
fo3.write(prefix)
for line in fo1:
    tags=line.strip().split("\t")
    mt_tags=copy.deepcopy(tags)
    asr_tags=copy.deepcopy(tags)
    mt_tags[0]=""
    mt_tags[1]=""
    mt_tags[2]=""
    asr_tags[3]=""
    mt_line='\t'.join(mt_tags) + '\n'
    asr_line='\t'.join(asr_tags) + '\n'
    fo2.write(mt_line)
    fo3.write(asr_line)
