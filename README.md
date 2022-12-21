# Capstone Project - Vegetables Classifications 
--- 
## Description of the Problem :
- From vegetable production to delivery, several common steps are operated manually. Like picking, and sorting vegetables. Therefore, we decided to solve this problem using deep neural architecture, by developing a model that can detect and classify vegetables. That model can be implemented in different types of devices and can also solve other problems related to the identification of vegetables, like labeling the vegetables automatically without any need for human work. (source - kaggle)

(For more information read this dataset on Kaggle - https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) 

Note - ( Steps to Download the Dataset from Kaggle are available in the notebook as well as
For more details, please read this -
https://www.kaggle.com/general/74235 )

**To solve this problem of vegetable classification, we used the CNN (Convolution Neural Network) based architecture eg. xcpetion architecture and also deployed the service to AWS lambda (serverless technology in the cloud)**

- Libaries/Language Used : Python3,  Numpy, Matplotlib, Tensorflow,Keras, Tensorflow Lite, flask, gunicorn, keras-image-helper etc
- Technologies/Cloud : Google Colab (or any other GPU enabled jupyter notebook provider ), AWS EC2 instance, AWS ECR (Amazon Elastic Container Registry), Docker, AWS Lambda.

---

## Instructions on how to run the project -
---
- Note - Preferred Environment (Linux Based - Ubuntu, Redhat, CentOS etc ), also we used the AWS Cloud for the deployment.
- Install Python >= 3.9 and pip for packages management
- On command line terminal use -
  
  - ```pip install -r requirements.txt```

Note - You can install the tensorflow for either CPU or GPU (please check this link for more details - https://www.tensorflow.org/install/pip)
Also, can install the tensorflow lite using this link-
(https://www.tensorflow.org/lite/guide/python)

Make sure you have all the necessary libraries installed as mentioned in ```requirements.txt```

- There is ```notebook.ipynb``` which you can explore to understand downloading the dataset from kaggle, EDA for image data, training the model using transfer learning, tuning the hyperparameters of the models, regularizations and dropout etc.

Note - After training the model, we saved it named ```xception_v4_1_10_0.998.h5``` (and accuracy was almost 99.8%), we will use this model for further.

- Now we convert the ```xception_v4_1_10_0.998.h5``` keras model into ```tflite``` format which is lightweight and used only for inference and deployment on the cloud. For converting the keras model from ```.h5``` format to ```.tflite``` format, we use the notebook ```tensorflow-lite-model.ipynb``` and convert the model as ```vegclass-model.tflite``` which we will use further for deployment.
- Now we create the Flask app (using gunicorn production ready webserver). So run the ```predict.py``` file using ```gunicorn --bind 0.0.0.0:9696 predict:app``` and the server run on ```localhost``` on port ```9696``` .
- Open another terminal and run the ```test.py``` file using ```python test.py``` and we see the results/predcition of image classification by our model running as webservice.
![gunicron webservice](https://user-images.githubusercontent.com/32613450/208980844-a3bc9245-f270-4cad-b3da-18a92cb0f298.png)
![response testpy](https://user-images.githubusercontent.com/32613450/208981095-37fd2371-a885-486a-a318-e01875b31ef5.png)


-- Now we have our model running as web service locally, but we will deploy it on cloud using AWS Lambda and for that we need to package all the libraries and files in the ```Dockerfile``` 

---
## To Run/Deploy the Project Locally on docker container -

1 ) Make sure your system has python > 3.9 version installed

2 ) Docker installed and running/active

3 ) Git installed

4 ) Clone the repository using --> ```git clone https://github.com/modfa/Capstone-Vegetable-Classification.git```

5 ) ```cd Capstone-Vegetable-Classification```

6 ) To build the image for running the container

```docker build -t projectimage .```

7 ) Command to run the docker container-
 ```docker run -it -p 8080:8080 --rm projectimage:latest```

8 ) Update the test.py file and change it to localrun on local host/uncomment the line (read the file for more)

9 ) Now run the ```test.py``` file from your local system using ```python test.py```  and you will see the prediciton from the model which has been deployed locally using the image for AWS lambda.

-- Now the Image build using the ```Dockerfile``` running locally is working fine and tested properly, now we can use this image and deploy it on AWS Lambda.

---
## Uploading the image to AWS ECR -
- We will us the ```aws cli``` for uploading the the docker image to AWS ECR repository.
-  To install the aws cli use this -  
```pip install awscli```

- we need to configure the ```awscli``` on our local computer (Check link - https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
- Make sure ```awscli``` is installed and configured using your credentials.
- now we create the ecr empty repository locally using -  
```aws ecr create-repository --repository-name vegclass-tflite-images```

It will return the URI for the repository, note it down.

- note down the repository uri 

```ACCOUNT=xxxxxxxxx```

```REGION=ap-south-1```

```REGISTRY=vegclass-tflite-images```

```PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}```

```TAG=vegclass-model-xception-v4-001```

```REMOTE_URI=${PREFIX}:${TAG}```

- now we need to login into the ecr registory as it is only available for the aws account holder (this will return the password for the ecr repository) using this command - 
     ```aws ecr get-login --no-include-email```


- now we need to execute the command which was return by previous one    
   ```$(aws ecr get-login --no-include-email)```

- it will print the successful login message

- now we will tag the previously tested and created docker image with the above ```REMOTE_URI```                  

```docker tag vegclass-model:latest ${REMOTE_URI}```
  
- now we can push the created docker image to the ecr registory   
     ```docker push ${REMOTE_URI}```

- now we can check the aws ecr section using the AWS graphical/website console that the image has been successfully pushed to ecr repository
![ecr repo](https://user-images.githubusercontent.com/32613450/208981314-71f301bd-1645-418c-8fbf-3eb51a567e56.png)
![ecr repo console](https://user-images.githubusercontent.com/32613450/208981349-a25014d8-b43c-48ea-a14e-ed4bb9722926.png)


--- 
## Deploying the Docker Image from ECR using the AWS Lambda

- now we can use this image(pushed to ECR) for our lambda function. We will use the web console for using the lambda function.
- go to AWS Lambda ---> Create Function ,
See the screenshots below 
creating the lambda function
![lambda function choose container](https://user-images.githubusercontent.com/32613450/208981527-692638e8-d605-4e94-96f2-0d37525f6d99.jpg)

- we need to change the default configurations
Edit -> increase the memory to 1024 or more and timeout to 30 seconds

![select configuration](https://user-images.githubusercontent.com/32613450/208981609-7fcef579-5ccb-4fd3-9122-a67138a97f34.png)
![edit config of lambda](https://user-images.githubusercontent.com/32613450/208981719-b083c60d-49b5-4a12-a669-65eb3baa3a21.png)

- Now go to test and create an test event
after the setup, we can test the event and see the same results which were available for us for local setup
![lambda test creation](https://user-images.githubusercontent.com/32613450/208981845-baf4ac29-8122-4d31-8f33-32628e10c273.png)
![json event test](https://user-images.githubusercontent.com/32613450/208981872-dba02bcf-ecf5-495f-9dbf-6da3154d8e4a.png)
![test success response](https://user-images.githubusercontent.com/32613450/208981925-f9863cc2-a956-47c1-8b2a-e6a73c268de4.png)


- now we can use the created lambda fucntion and expose it as a web service using API gateway

- go to API Gateway --> REST API
Choose new api --> fill the name --> create api (See screenshots)
![api gateway console first](https://user-images.githubusercontent.com/32613450/208982012-4caa89df-686d-49ee-a26d-25aea7d0bf44.png)


- go to actions --> create resources --> set the endpoint name as predict and create it

![api gateway create resource](https://user-images.githubusercontent.com/32613450/208982194-8ffc730b-167c-4698-9161-8489823f8149.png)
![api gateway resource2](https://user-images.githubusercontent.com/32613450/208982230-1937c5af-50c0-41d2-9e3d-8d0745ac8bde.png)

- now again choose the actions --> select the method (POST)
![api gate post method setup](https://user-images.githubusercontent.com/32613450/208982334-e7a53a8f-6ecd-4fbd-8710-e737eb4b997b.png)

now choose the lambda function (select the earlier created lambda function ) and save it


- now grant the permission of gateway service to invoke the lambda function
![permission for gate to lambda](https://user-images.githubusercontent.com/32613450/208982481-2b2d49e1-41db-461d-a10c-7c0decb6fec6.png)


- now it will give a visual representation and choose the test
![test visual](https://user-images.githubusercontent.com/32613450/208982537-25a977af-21bd-4e42-bedf-20b27db7799a.png)


- click on test --> and put the json in request body


- now we can see that it respond with the result of classification from our model which was deployed using the lambda function

- now we can take this api gateway and deploy it (or expose it) , create new stage (name it anything) and click deploy

- now it will give us the url which we can use to test the api gateway

- now we can use our ```test.py``` script created earlier and modify the given url( https://itboq3h6ob.execute-api.ap-south-1.amazonaws.com/test/predict) to test (uncomment the url as mentioned in the ```test.py``` script)
- the above api gawateway method will invoke the lambda fucntion and provide us the results.

- now using the terminal we can test our test.py script and it goes to api gateway which in turn invoke the lambda fucntion and it returns the response to api gateway and pai gawateway to our local client
![response testpy](https://user-images.githubusercontent.com/32613450/208982757-a48ba312-ca47-4bad-999a-004d91995613.png)


- this is how we turn our lambda function to a web service using the api gateway.
