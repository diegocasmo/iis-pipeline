import sys
import random
from furnet import *
import json
   
furnet = Furnet('localhost', 1337)
		
furnet.connect()

if furnet.connected:
    #furnet.say('bla bla')
    #furnet.gaze(1.0, 1.0, 1.0, True)

    #precision = {'anger': 58.6, 'surprise': 58.6, 'hapiness': 40.0, 'sadness': 30, 'disgust': 20, 'fear': 55, 'emotion_neutral': 10}
    emos = ['anger', 'surprise', 'hapiness', 'sadness', 'disgust', 'fear'] #'emotion_neutral'

    with open('output.json') as json_data:
        precision = json.load(json_data)
      
#pipeline - reaction according to recived input -----
    bigger=-1.0
    emotion='emotion_neutral'
    for e, p in precision.items():
        print 'e: ',e, ' p:', p, ' bigger:', bigger
        if p > bigger:
           bigger = p
           emotion=e
        elif p == bigger:
           emotion = 'emotion_neutral' #more than 2 emos with similar precisions
    print 'The response emotion:', emotion, ' confidence score:', p
    furnet.gesture(emotion)
    '''
    #----------------------------------------------------
    #print len(sys.argv)       
    if len(sys.argv) > 1:
        emotion=sys.argv[1]    
        #furnet.gesture('emotion_neutral')
        furnet.gesture(emotion)
#generates randomly 6 emos, waits for enter key-----          
    else:
        for i in range(len(emos)):
            #furnet.gesture('emotion_neutral')
            emotion = random.choice(emos)
            emos.remove(emotion)
            furnet.gesture(emotion) #mimics or confuse
            raw_input('Press enter to continue: ')
            furnet.gesture('emotion_neutral')
            raw_input('Press enter to continue: ')           
    #----------------------------------------------------
    '''            
    print "Waiting for executed ack responses",
    sys.stdout.flush()
    
    # We should get 3 elements back when all of those 
    # commands are executed, while waiting we sleep
    while True:
        if furnet.receive_queue.qsize() > 0:
            print "Received", furnet.receive_queue.get()
        else:
            break
    #furnet.gesture('emotion_neutral')
    furnet.close()
