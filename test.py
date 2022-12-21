import requests

# uncomment this line if you are running the container (built using Dockerfile)
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'


#uncomment this line if you are running the conatiner on AWS Lambda (Serverless)
# url = ' https://itboq3h6ob.execute-api.ap-south-1.amazonaws.com/test/predict'


#uncomment this line while testing the webservice running locally using the flask/gunicorn
# url = 'http://localhost:9696/predict'


data = { 'url' : 'https://i.postimg.cc/LXrRH5xP/capsicum.jpg' }

result = requests.post(url, json=data).json()
print(result)