fo1 = open("dict","r")
fo2 = open("dict.out","w")

for line in fo1:
    tags=line.split(" ")
    fo2.write(tags[0]+" 1\n")
