

import tensorflow as tf
import subprocess
import re
import numpy as np
import mltk

PROFILE_ERROR_FLAG = 0


def get_ondevice_hardware_attributes(m,ds):
        
        def representative_dataset():
            for input_value in tf.data.Dataset.from_tensor_slices(ds[0]).batch(1).take(100):
                yield [tf.dtypes.cast(input_value, tf.float32)]

       

        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        #converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()

        with open("test.tflite", 'wb') as f:
            f.write(tflite_quant_model)

        command = "mltk profile test.tflite"
        log = ""
        # Execute the command in the system shell and capture output
        try:
            completed_process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            log = completed_process.stdout
            #print(log)

        except subprocess.CalledProcessError as e:
            print("Error:", e)

        return log

def parse_ondevice_hardware_attributes(log):

    global PROFILE_ERROR_FLAG

    ram_value = re.search(r'RAM, Runtime Memory Size \(bytes\): (\d+\.\d+)(k|M)', log)
    flash_value = re.search(r'Flash, Model File Size \(bytes\): (\d+\.\d+)(k|M)', log)
    mac_value = re.search(r'Multiply-Accumulate Count: (\d+\.\d+)(k|M)', log)

    
    if PROFILE_ERROR_FLAG == 0 :
        ram = -1
        flash = -1
        macc = -1

    else :

        ram =float("inf")
        flash = float("inf")
        macc = float("inf")

    if ram_value:
        ram_size, ram_unit = ram_value.group(1), ram_value.group(2)
        ram = float(ram_size)
        if ram_unit == 'M':
            ram *= 1000000
        elif ram_unit == 'k':
            ram *= 1000

    if flash_value:
        flash_size, flash_unit = flash_value.group(1), flash_value.group(2)
        flash = float(flash_size)
        if flash_unit == 'M':
            flash *= 1000000
        elif flash_unit == 'k':
            flash *= 1000

    if mac_value:
        mac_size, mac_unit = mac_value.group(1), mac_value.group(2)
        macc = float(mac_size)
        if mac_unit == 'M':
            macc *= 1000000
        elif mac_unit == 'k':
            macc *= 1000


    if (ram != -1) or (flash != -1) :
        PROFILE_ERROR_FLAG = 1
        
        
    return ram, flash, macc 

def evaluate_hardware_requirements(model, ds):
    log = get_ondevice_hardware_attributes(model,ds)
    return parse_ondevice_hardware_attributes(log)
