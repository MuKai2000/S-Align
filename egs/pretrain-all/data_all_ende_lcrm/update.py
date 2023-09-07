import sys

fo1 = open("train_st.tsv","r")
fo2 = open("dev_st.tsv","r")
fo3 = open("tst-COMMON_st.tsv","r")
out1 = open("train_st.update.tsv","w")
out2 = open("dev_st.update.tsv","w")
out3 = open("tst-COMMON_st.update.tsv","w")

out1.write(fo1.readline())
out2.write(fo2.readline())
out3.write(fo3.readline())
prefix1="/mnt/zhangyuhao/MSP-ST/mustc/train/wav-split/"
prefix2="/mnt/zhangyuhao/MSP-ST/mustc/dev/wav-split/"
prefix3="/mnt/zhangyuhao/MSP-ST/mustc/tst-COMMON/wav-split/"
for line in fo1:
    tags=line.split("\t")
    tags[0]="train_"+tags[0]
    tags[1]=prefix1+tags[0]+".wav"
    new_line='\t'.join(tags)
    out1.write(new_line)

for line in fo2:
    tags=line.split("\t")
    tags[0]="dev_"+tags[0]
    tags[1]=prefix2+tags[0]+".wav"
    new_line='\t'.join(tags)
    out2.write(new_line)

for line in fo3:
    tags=line.split("\t")
    tags[0]="test_"+tags[0]
    tags[1]=prefix3+tags[0]+".wav"
    new_line='\t'.join(tags)
    out3.write(new_line)
