import requests
from random import randint,random
import json
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# import sys
# sys.path.append("../../../notebooks")
# from visualizer import get_graph
# from seldon_core.proto import prediction_pb2
# from seldon_core.proto import prediction_pb2_grpc
# import grpc
# import tensorflow as tf
# from tensorflow.core.framework.tensor_pb2 import TensorProto
from seldon_core.seldon_client import SeldonClient
import time

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d,cmap=plt.cm.gray_r, interpolation='nearest')
    return plt

def download_mnist():
    return input_data.read_data_sets("MNIST_data/", one_hot = True)

# def rest_predict_request(endpoint,data):
#     request = {"data":{"ndarray":data.tolist()}}
#     response = requests.post(
#                 "http://"+endpoint+"/predict",
#                 data={"json":json.dumps(request),"isDefault":True})
#     return response.json()   

# def rest_transform_input_request(endpoint,data):
#     request = {"data":{"ndarray":data.tolist()}}
#     response = requests.post(
#                 "http://"+endpoint+"/transform-input",
#                 data={"json":json.dumps(request),"isDefault":True})
#     return response.json()   

# def rest_transform_output_request(endpoint,data):
#     request = {"data":{"ndarray":data.tolist()}}
#     response = requests.post(
#                 "http://"+endpoint+"/transform-output",
#                 data={"json":json.dumps(request),"isDefault":True})
#     return response.json()   

# def rest_request_ambassador(deploymentName,namespace,endpoint="localhost:8003",arr=None):
#     payload = {"data":{"names":["a","b"],"tensor":{"shape":[1,784],"values":arr.tolist()}}}
#     response = requests.post(
#                 "http://"+endpoint+"/seldon/"+namespace+"/"+deploymentName+"/api/v0.1/predictions",
#                 json=payload)
#     print(response.status_code)
#     print(response.text)

# def grpc_request_internal(data,endpoint="localhost:5000"):
#     datadef = prediction_pb2.DefaultData(
#             tftensor=tf.make_tensor_proto(data)
#         )

#     request = prediction_pb2.SeldonMessage(data = datadef)
#     channel = grpc.insecure_channel(endpoint)
#     stub = prediction_pb2_grpc.ModelStub(channel)
#     response = stub.Predict(request=request)
#     return response


def gen_mnist_data(mnist):
    batch_xs, batch_ys = mnist.train.next_batch(1)
    chosen=0
    # gen_image(batch_xs[chosen]).show()
    data = batch_xs[chosen].reshape((1,784))
    return data


mnist = download_mnist()
data = gen_mnist_data(mnist)
# data = data.reshape((784))
time1 = time.time()
sc = SeldonClient(deployment_name="tfserving-default",namespace="default",gateway_endpoint="146.148.42.77",gateway="seldon")
time2 = time.time()
r = sc.predict(transport="rest", data=data)
time3 = time.time()
print(r)

print(time3 - time2)
print(time3 - time1)

# rest_request_ambassador("tfserving-mnist","seldon",endpoint="146.148.42.77/one",arr=data)
# rest_predict_request(endpoint="http://10.48.2.35",data=data)