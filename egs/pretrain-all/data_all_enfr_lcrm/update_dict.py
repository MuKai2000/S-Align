import sys
fo1 = open(sys.argv[1],"r")
fo2 = open("dict.wrd.out","w")

for line in fo1:
    tags=line.split(" ")
    fo2.write(tags[0]+" 1\n")
