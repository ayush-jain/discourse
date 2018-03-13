import csv
import pickle


with open('train_data.pkl', 'rb') as handle:
    train_dict = pickle.load(handle)    

with open('dev_data.pkl', 'rb') as handle:
    dev_dict = pickle.load(handle)    


with open('test_data.pkl', 'rb') as handle:
    test_dict = pickle.load(handle)
    
#print(len(train_dict[53971]))    
'''
for sdid,lis in train_dict.iteritems():
        res = []        
        #print(len(lis))
        print("Thread-" + str(sdid))
        for each in lis:
            print(each[0]['url'])
            print("\n")
            print(each[0]['headline'])
            print("-------------------")            
            for cmtinx, comment_dict in each.iteritems():
                #print(len(comment_dict.values()))
                #print(comment_dict)
                print(str(cmtinx) + ":-" + comment_dict['text'].strip())
            
            break            
            
        print("\n\n\n")

print(len(train_dict.keys()))        


for sdid,lis in dev_dict.iteritems():
        res = []        
        #print(len(lis))
        print("Thread-" + str(sdid))
        for each in lis:
            print(each[0]['url'])
            print("\n")
            print(each[0]['headline'])
            print("-------------------")            
            for cmtinx, comment_dict in each.iteritems():
                #print(len(comment_dict.values()))
                #print(comment_dict)
                print(str(cmtinx) + ":-" + comment_dict['text'].strip())
                    
            break            
        print("\n\n\n")

print(len(dev_dict.keys()))        
'''

for sdid,lis in test_dict.iteritems():
        res = []        
        #print(len(lis))
        print("Thread-" + str(sdid))
        for each in lis:
            print(each[0]['url'])
            print("\n")
            print(each[0]['headline'])
            print("-------------------")            
            for cmtinx, comment_dict in each.iteritems():
                #print(len(comment_dict.values()))
                #print(comment_dict)
                print(str(cmtinx) + ":-" + comment_dict['text'].strip())
                    
            break    
        print("\n\n\n")

print(len(test_dict.keys()))        


                       
