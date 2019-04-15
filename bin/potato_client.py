#!/usr/bin/env python3
import grpc
import potato_pb2
import potato_pb2_grpc
import socket

import pandas as pd

# input: features, Pandas.DataFrame
# output: hint, docker_image  
def get_hint(features):
    myhostname = socket.gethostname()
    
    channel = grpc.insecure_channel('localhost:50051')
    
    stub = potato_pb2_grpc.HintStub(channel)
        
    request = potato_pb2.HintRequest(hostname=myhostname) 
        
    print('Wait for response from POTATO server...')
    response = stub.Hint(request)

    if len(features) > 0:
        stub = potato_pb2_grpc.HintStub(channel)
        request = potato_pb2.HintRequest(hostname=myhostname,
                                          active_cpu_ratio=43) 
        response = stub.Hint(request)
        hint = response.hint 
        docker_image = response.docker_image
    else:
        hint = 'There is no features to get hints.' 
        docker_image = 'NA' 

    return hint, docker_image 

     
