import gradio as gr
from fastai.vision.all import *
import skimage

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPaths

learn = load_learner('truck_classifier_v2.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Truck Classifier"
description = "A truck classifier trained on internet image dataset with fastai."
article="<p style='text-align: center'>Thank you</p>"
examples = []#['fire_truck.jpg', 'freight_truck.jpg', 'pickup_truck.jpg', 'tanker_truck.jpg']
enable_queue=True

gr.Interface(fn=predict,inputs=gr.Image(shape=(512, 512)),
             outputs=gr.Label(num_top_classes=3),title=title,
             description=description,article=article,examples=examples).launch(
    enable_queue=enable_queue)