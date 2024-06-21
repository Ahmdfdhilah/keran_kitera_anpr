import os

# Mendapatkan path direktori saat ini
current_directory = os.getcwd()

# Google Drive
model_file_id = "1qXceiWK0oMClrfPXkBhfIvjlcS0nuJKz"
model_file_name = "darknet-yolov3.cfg"
weights_file_name = "lapi.weights"
weights_file_id = "18e7Cu5hs6IdZoxtjnJjuq_qodJKTyMol"

classes_file = os.path.join(current_directory, "config", "classes.names")
model_configuration = os.path.join(current_directory, "models", model_file_name)
model_weights = os.path.join(current_directory, "models", weights_file_name)
result_path = os.path.join(current_directory, "result")
conf_threshold = 0.95
nms_threshold = 0.4
inp_width = 416
inp_height = 416
