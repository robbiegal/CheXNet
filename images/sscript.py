factor=13
data_orig=open("test_list_short.txt",'r').read().split('\n')
data_new=[]
for i in range(len(data_orig)):
    if not i%factor:
        data_new.append(data_orig[i])

data_new_writer=open('test_list_factor_'+str(factor)+'.txt','wr').write('\n'.join(data_new))

