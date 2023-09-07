fo=open("tgt","r")
lines=fo.read().strip.split("\n")
cot=0
for line in lines:
    cot+=len(line.strip().split(" "))
print(cot)
