import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model(r'c:\Users\userr\Desktop\INZY\models\field_classifier_finetuned.keras')
print("MODEL SUMMARY")
model.summary()

print("\\nLAYERS AND WEIGHT STATS")
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if not weights:
        continue
    # flatten all weight arrays for this layer
    flat = np.concatenate([w.ravel() for w in weights])
    print(f"#{i} {layer.name:30s} | type={layer.__class__.__name__:12s} | params={flat.size:6d} | mean={flat.mean(): .6f} std={flat.std(): .6f} min={flat.min(): .6f} max={flat.max(): .6f}")