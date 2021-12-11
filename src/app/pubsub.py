from google.cloud import pubsub_v1

import os

def publish_new_score(msg):
    project_id = os.environ.get('GCP_PROJECT')
    topic_id = "topic-new-score-created"
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    data = msg.encode('utf-8')
    future = publisher.publish(topic_path, data)
    
    print(future.result())
    print('Published message to: ', topic_path)    
    return future
    
if __name__ == '__main__':
    publish_new_score('{"cpf":1234455,\
        "request_datetime":"2021-01-01",\
            "score":750, "status":"APROVADO"}')    
    