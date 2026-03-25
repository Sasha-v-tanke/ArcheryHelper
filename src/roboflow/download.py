from roboflow import Roboflow

rf = Roboflow(api_key="dtHqxiRwXR8yMvf0Rp6e")
project = rf.workspace("uni-oidi4").project("archery-e6v5r")
model = project.version(1).download("yolov8")
