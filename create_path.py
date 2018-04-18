import sys
import glob

'''
The code helps to generate the train.txt and test.txt file 
needed for yolo training. 
'''

arg = (sys.argv)
train_path = arg[1]
test_path  = arg[2]
file_ext   = arg[3]


with open('train.txt','w') as f:
	tp = train_path + '*.' + file_ext
	print tp
	for i in glob.glob(tp):
		
		f.write(i+'\n')


with open('test.txt','w') as f:
	tp = test_path + '*.' + file_ext
	print tp
	for i in glob.glob(tp):
		
		f.write(i+'\n')


