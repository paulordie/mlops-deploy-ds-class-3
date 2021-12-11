import base64
import json

def initial_method(event, context):
    #Decode pubsub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    score_message = json.loads(pubsub_message)
    print("Received message: ",str(score_message))
    
