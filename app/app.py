import os
import base64
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
from io import BytesIO

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "final_model.keras")
