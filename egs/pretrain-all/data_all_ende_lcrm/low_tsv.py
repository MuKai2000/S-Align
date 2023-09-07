import sys

fo1=open(sys.argv[1],"r")
fo2=open(sys.argv[1]+".lc","w")
fo2.write(fo1.readline())
for line in fo1:
    tags=line.strip().split("\t")
    tags[-1]=tags[-1].lower()
    if "asr" in sys.argv[1]:
        tags[-2] = tags[-1]
        
    new_line = '\t'.join(tags)+"\n"
    if "mt" in sys.argv[1]:
        new_line="\t"+"\t"+"\t"+new_line
    fo2.write(new_line)
