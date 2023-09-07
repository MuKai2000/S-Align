fo = open("src.token","r")

length=0
for line in fo:
    length = max(length, len(line.strip().split(" ")))
print(length)
