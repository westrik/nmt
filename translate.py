import pickle as pkl

try:
    with open('wordmap.pkl','r') as wordmap:
        source, dst = pkl.load(wordmap)
except:
    print "Could not open word map"
    exit(1)
