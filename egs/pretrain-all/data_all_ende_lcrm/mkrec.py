org = open("./dev_st.tsv", 'r')
data = org.readlines()
org.close()
for i in range(1,len(data)):
    data[i] = data[i].split('\t')
    data[i][-2] = data[i][-1].strip('\n')
    data[i] = '\t'.join(data[i])
print(data[:10])
fin = open("./dev_rec_st.tsv", "w")
fin.writelines(data)
fin.close()

