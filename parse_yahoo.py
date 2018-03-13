import csv
import pickle



train_ids = []
dev_ids = [] 
test_ids = []

#sdid-[{}, {}, {}]

train_dict = {}
dev_dict = {}
test_dict = {}

train_labels = {}
dev_labels = {}
test_labels = {}

file = open("ydata-ynacc-v1_0_train-ids.txt", "r") 
for line in file: 
    train_ids.append(int(line.strip()))
    
file = open("ydata-ynacc-v1_0_dev-ids.txt", "r") 
for line in file: 
    dev_ids.append(int(line.strip()))
    
file = open("ydata-ynacc-v1_0_test-ids.txt", "r") 
for line in file: 
    test_ids.append(int(line.strip()))
    
               

#open train, dev and test ids along with expert,turk 


def fetch_data(sample_ids, sample_file, data):
    global train_dict
    global dev_dict
    global test_dict
    
    if(data == 'train'):
        sample_dict = train_dict
    elif(data == 'dev'):
        sample_dict = dev_dict
    elif(data == 'test'):
        sample_dict = test_dict        
    
    #cmtinx-{}
    thread_dict = {}
    i=0
    prev = 0
    cmtlist = []
    with open(sample_file) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            
            if(i==0):
                header = line
            else:
                
                j=0
                    
                sdid = int(line[0])
                    
                if(sdid not in sample_ids):
                    continue
                
                if(i==1):
                    prev = sdid
                    sample_dict[sdid] = []
                
                cmtinx = int(line[1])
                #check if different thread has started or there are no 2 consecutive threads with same sdid
                if(sdid != prev or (cmtinx in cmtlist)):
                    if(sdid not in sample_dict):
                        sample_dict[sdid] = []
                    sample_dict[prev].append(thread_dict)        
                    thread_dict = {}
                    prev = sdid
                    cmtlist = []
                    
                cmtlist.append(cmtinx)    
                comment_dict = {}                    
                for each in line:
                    #print(each)
                    #if each == '':
                    #    continue
                    comment_dict[header[j]] = each              
                    j=j+1
                thread_dict[cmtinx] = comment_dict
                
                    
                #print(thread_dict)
                     
                    #print line    
                
            i=i+1
                       

expert_file = "ydata-ynacc-v1_0_expert_annotations.tsv"
turk_file = "ydata-ynacc-v1_0_turk_annotations.tsv"

fetch_data(train_ids, expert_file, 'train')

#print(len(train_dict[53971]))
#print(train_dict[53971])


fetch_data(train_ids, turk_file, 'train')

#print(len(train_dict[53971]))
#print(train_dict[53971])


fetch_data(dev_ids, expert_file, 'dev')
fetch_data(dev_ids, turk_file, 'dev')

fetch_data(test_ids, expert_file, 'test')
fetch_data(test_ids, turk_file, 'test')

print(len(train_dict))
print(len(dev_dict))
print(len(test_dict))



'''

#cmtinx-{}
thread_dict = {}
i=0
prev = 0
cmtlist = []
with open("ydata-ynacc-v1_0_turk_annotations.tsv") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        if(i==0):
            header = line
        else:
            #print("hello")
            j=0
            sdid = int(line[0])
            if(sdid not in test_ids):
                continue
            #print("hello")    
            cmtinx = int(line[1])
            #check if different thread has started or there are no 2 consecutive threads with same sdid
            if(sdid != prev or (cmtinx in cmtlist)):
                if(sdid not in test_dict):
                    test_dict[sdid] = []    
                test_dict[sdid].append(thread_dict)
                thread_dict = {}
                prev = sdid
                cmtlist = []
            cmtlist.append(cmtinx)    
            comment_dict = {}                    
            for each in line:
                if each == '':
                    continue
                comment_dict[header[j]] = each              
                j=j+1
            thread_dict[cmtinx] = comment_dict     
            print(thread_dict[0])
                #print line    
            
        i=i+1




print(i)
print((thread_dict))
#print(len(train_dict[84848]))
print(len(test_dict))        


'''

with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_dict, f)  
    
with open('dev_data.pkl', 'wb') as f:
    pickle.dump(dev_dict, f)
    
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_dict, f)    


print(len(train_dict[53971]))
#print(train_dict[53971][0][0]['sdid'])
#print(train_dict[53971][1][0]['sdid'])
#print(train_dict[53971][2][0]['sdid'])
#print(train_dict[53971][3][0]['sdid'])
#print(train_dict[53971][4][0]['sdid'])

with open('train_data.pkl', 'rb') as handle:
    train_dict = pickle.load(handle)    

with open('dev_data.pkl', 'rb') as handle:
    dev_dict = pickle.load(handle)    


with open('test_data.pkl', 'rb') as handle:
    test_dict = pickle.load(handle)


def fetch_labels(sample_dict, sample_labels):
    constructive = 0
    sd_type  = ''
    type_bool = 0
    count = 0        
    for sdid,lis in sample_dict.iteritems():
        res = []    
        #print(len(lis))
        for each in lis:            
            for cmtinx, comment_dict in each.iteritems():
                #print(len(comment_dict.values()))
                #print(comment_dict)
                for key,val in comment_dict.iteritems():
                    #print key
                    #print val
                    if (key =='constructiveclass'):
                        if(val == 'Constructive'):
                            constructive = 1
                        else:
                            constructive = 0
                            
                    
                    if(key == 'sd_type'):
                        sd_type = val
                        if('Argumentative (back and forth)' in sd_type or 'Positive/respectful' in sd_type or 'Personal stories' in sd_type or 'Snarky/humorous' in sd_type):
                            type_bool = 1
                        else:   
                            type_bool = 0        
                                
            if(constructive == 0):
                res.append(0)
            if(constructive == 1 and type_bool == 1):
                res.append(1)
            elif(constructive == 1 and type_bool == 0):
                res.append(0)
                                    
            #print("************************\n\n\n\n\n\n\n") 
        
        #print(sdid)
        #print(res)
        
        #in case threads overalap between expert and turk files (3+3), priority to expert file
        if(len(res)>3):
            res = res[:3]
        #print(len(res))
        
        if(sum(res) <= 1 and len(res) == 3):
            sample_labels[sdid] = 0
            count = count+1
        elif(sum(res) > 1 and len(res) == 3):
            sample_labels[sdid] = 1
        elif(sum(res) == 1 and len(res) == 2):
            sample_labels[sdid] = 1
            #count = count+1
        else:    
            sample_labels[sdid] = 0
        #print("\n")
    
    print(count)         

fetch_labels(train_dict, train_labels)

fetch_labels(dev_dict, dev_labels)            

fetch_labels(test_dict, test_labels)
            
print("\n")
print(len(train_labels))
#no of 1s
print(sum(train_labels.values()))
print("\n")

print(len(dev_labels))
print(sum(dev_labels.values()))
print("\n")

print(len(test_labels))
print(sum(test_labels.values()))
print("\n")


with open('train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f)        

with open('dev_labels.pkl', 'wb') as f:
    pickle.dump(dev_labels, f)        

with open('test_labels.pkl', 'wb') as f:
    pickle.dump(test_labels, f)        
                                                                                          
