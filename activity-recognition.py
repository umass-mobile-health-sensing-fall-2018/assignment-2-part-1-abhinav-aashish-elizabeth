# -*- coding: utf-8 -*-
"""
This Python script receives incoming unlabelled accelerometer data through 
the server and uses your trained classifier to predict its class label.

"""

import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import extract_features # make sure features.py is in the same directory
from util import reorient, reset_vars

# TODO: Replace the string with your user ID
user_id = "aashish7k5"
# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["sitting", "walking", "jumping", "running"] #...


# Loading the classifier that you saved to disk previously
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
def onActivityDetected(activity):
    """
    Notifies the user of the current activity
    """
    print("Detected activity:" + activity)

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label. 
    """
    
    # TODO: extract features over the window of data
    feature_names, feature_vector = extract_features(window)
    
    # TODO: use classifier.predict(feature_vector) to predict the class label.
    # Make sure your feature vector is passed in the expected format
    classifier.predict(feature_vector)
    
    # TODO: get the name of your predicted activity from 'class_names' using the returned label.
    # pass the activity name to onActivityDetected()
    onActivityDetected(feature_names[feature_vector])    
    
    return
    

#################   Server Connection Code  ####################

'''
    This socket is used to receive data from the data collection server
'''
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receive_socket.connect(("none.cs.umass.edu", 8888))
# ensures that after 1 second, a keyboard interrupt will close
receive_socket.settimeout(1.0)

msg_request_id = "ID"
msg_authenticate = "ID,{}\n"
msg_acknowledge_id = "ACK"

def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.
    
    If it fails, it will raise an appropriate exception.
    """
    message = sock.recv(256).strip().decode('ascii')
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))
    sock.send(msg_authenticate.format(user_id).encode('utf-8'))

    try:
        message = sock.recv(256).strip().decode('ascii')
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")
        
    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception("Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))
    
    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))


try:
    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)
    
    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()
        
    previous_json = ''
    
    sensor_data = []
    window_size = 25 # ~1 sec assuming 25 Hz sampling rate
    step_size = 25 # no overlap
    index = 0 # to keep track of how many samples we have buffered so far
    reset_vars() # resets orientation variables
        
    while True:
        try:
            message = receive_socket.recv(1024).strip().decode('ascii')
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            print(json_strings[0])
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = '' # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == u"SENSOR_ACCEL"):
                    t=data['data']['t']
                    x=data['data']['x']
                    y=data['data']['y']
                    z=data['data']['z']
                        
                    sensor_data.append(reorient(x,y,z))
                    index+=1
                    # make sure we have exactly window_size data points :
                    while len(sensor_data) > window_size:
                        sensor_data.pop(0)
                
                    if (index >= step_size and len(sensor_data) == window_size):
                        t = threading.Thread(target=predict, args=(np.asarray(sensor_data[:]),))
                        t.start()
                        index = 0
                
            sys.stdout.flush()
        except KeyboardInterrupt: 
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            break
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely       
                print(e)
            pass
except KeyboardInterrupt: 
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Qutting...")
finally:
    print('closing socket for receiving data')
    receive_socket.shutdown(socket.SHUT_RDWR)
    receive_socket.close()