
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor



interpreter = tflite.Interpreter(model_path='vegclass-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']




preprocessor = create_preprocessor('xception', target_size=(299,299))


# url = 'https://i.postimg.cc/LXrRH5xP/capsicum.jpg'

classes = [
    'Bean',
    'Bitter_Gourd',
 'Bottle_Gourd',
 'Brinjal',
 'Broccoli',
 'Cabbage',
 'Capsicum',
 'Carrot',
 'Cauliflower',
 'Cucumber',
 'Papaya',
 'Potato',
 'Pumpkin',
 'Radish',
 'Tomato']


def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result





