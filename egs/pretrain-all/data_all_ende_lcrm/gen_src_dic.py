fo1 = open("src.token","r")
fo2 = open("src.dict","w")

d={}

for line in fo1:
    tokens=line.strip().split(" ")
    for token in tokens:
        if token not in d.keys():
            d[token] = 1
        else:
            d[token] += 2
    
d=sorted(d.items(), key=lambda x: x[1], reverse=True)
for i in d:
    fo2.write(i[0]+" "+str(i[1])+"\n")
