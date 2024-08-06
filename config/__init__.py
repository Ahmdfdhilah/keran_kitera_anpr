import os

# Mendapatkan path direktori saat ini
current_directory = os.getcwd()

# Google Drive
model_file_id = "1qXceiWK0oMClrfPXkBhfIvjlcS0nuJKz"
weights_file_id = "18e7Cu5hs6IdZoxtjnJjuq_qodJKTyMol"
model_file_name = "yolov4-ANPR.cfg"
# model_file_name = "yolov8n.pt"
weights_file_name = "yolov4-ANPR.weights"
names_file_name = "yolov4-ANPR.names"

classes_file = os.path.join(current_directory, "models", names_file_name)
model_configuration = os.path.join(current_directory, "models", model_file_name)
model_weights = os.path.join(current_directory, "models", weights_file_name)
result_path = os.path.join(current_directory, "result")
conf_threshold = 0.9
nms_threshold = 0.3
inp_width = 416
inp_height = 416
