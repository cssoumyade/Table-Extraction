import os
import re
import numpy as np
import csv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from time import strftime
import pytesseract
import tensorflow as tf

def load_interpreter(model_path=None):
    """
    This function loads a tflite model interpreter 
    """
    if model_path is None:
        model_path = 'tablenet_densenet121_lite.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter 

def adjust(new_rows, maxi):
    """
    A function to set all with maxi number of columns
    for making csv compatible
    """
    rows = []
    for each_row in new_rows:
        if len(each_row) < maxi:
            for i in range(maxi - len(each_row)):
                each_row.append("-")
        rows.append(each_row)
    return rows

def text2csv(text):
    """
    This funtion transorms a text with newline and spaces to
    a csv that treats the spaces in the text as comma and newlines as carriage return
    """
    rows = text.split('\n')
    new_rows = []
    maxi = 0
    for each_row in rows:
        temp_row = each_row.split()
        if maxi < len(temp_row):
            maxi = len(temp_row)
        new_rows.append(temp_row)
    new_rows = adjust(new_rows, maxi)
    header = ['column_{}'.format(i) for i in range(maxi)]
    tstr = strftime("%Y%m%d-%H%M")
    temp_dir = 'output'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_file = os.path.join(temp_dir, 'temp_{}.csv'.format(tstr))

    with open(temp_file, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(new_rows)
    return temp_file

def append_offset(name, offset):
    """
    This function is used for assigning a name with offset if a file with the same name exists
    It takes a filename and a offset and returns a valid equivalent name with offset number
    
    Example :
    # assume two variables 
    name = 'python.py'
    offset = '2'
    append_offset(name, offset)
    
    # The above invocation will return string as
    # 'python_2.py'
    """
    fname, extension = name.split('.')
    fname = ''.join([fname, '_', offset, '.', extension])
    return fname

def render(mask):
  mask = tf.argmax(mask, axis=-1)
  mask = mask[..., tf.newaxis]
  return mask[0]

def visualize(image):
    plt.figure(figsize=(15, 15))
    title = 'Cropped Table'
    plt.title(title)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
    plt.axis('off')
    plt.show()


def final(img_path, output_dir='output', show_table=False):
    
    interpreter = load_interpreter()
    image_orig = Image.open(img_path)
    original_dim = image_orig.size
    image = image_orig.resize((512,512))
    np_image = np.asarray(image)/255.0
    np_image = np_image.astype(np.float32)
    np_image = np.expand_dims(np_image, axis=0)

    ip_d = interpreter.get_input_details()[0]
    op_d = interpreter.get_output_details()[0]

    interpreter.set_tensor(ip_d['index'], np_image)
    interpreter.invoke()
    
    tab_mask = interpreter.get_tensor(op_d['index'])

    
    tab_mask = np.squeeze(render(tab_mask).numpy())

    tab_mask = Image.fromarray(np.uint8(tab_mask))
    tab_mask = tab_mask.resize(original_dim)

    tab_mask = np.array(tab_mask)


    image_orig = np.array(image_orig)
    x, y, w, h = cv2.boundingRect(tab_mask)
    tab = image_orig[y:y+h, x:x+w]
    
    text = pytesseract.image_to_string(tab)
    text = text.strip()
    text = re.sub("[\r\n]+", "\r\n", text)
    csv = text2csv(text)
    if show_table:
       return csv, tab
    return csv