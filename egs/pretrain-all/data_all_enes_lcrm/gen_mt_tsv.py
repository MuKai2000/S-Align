src=open("train.en","r")
tgt=open("train.es","r")
out=open("train_wmt_mt.tsv","w")
out.write("id\taudio\tn_frames\ttgt_text\tsrc_text\n")
for a,b in zip(src,tgt):
    tags= ["","","",a.strip(),b.strip()]
    new_line="\t".join(tags)+"\n"
    out.write(new_line)
