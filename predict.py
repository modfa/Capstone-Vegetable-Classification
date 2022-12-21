
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from flask import Flask, request, jsonify



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


app = Flask('vegclassify')


@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.get_json()
    X = preprocessor.from_url(image_url['url'])
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return jsonify(dict(zip(classes, float_predictions)))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)