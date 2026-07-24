import os
import sys

# Make the "models" directory importable
sys.path.append("models")

from keras.models import load_model

from nfp.layers import (
    Squeeze,
    GatherAtomToBond,
    ReduceBondToAtom,
    ReduceAtomToPro
)
from nfp.models import GraphModel

from cascade.apply import predict_NMR_C


# -------------------------------------------------------------------
# Load trained 13C model
# -------------------------------------------------------------------
MODEL_PATH = os.path.join(
    "models",
    "cascade",
    "trained_model",
    "best_model.hdf5"
)

print("Loading model...")

model = load_model(
    MODEL_PATH,
    custom_objects={
        "GraphModel": GraphModel,
        "Squeeze": Squeeze,
        "GatherAtomToBond": GatherAtomToBond,
        "ReduceBondToAtom": ReduceBondToAtom,
        "ReduceAtomToPro": ReduceAtomToPro
    }
)

print("✓ Model loaded successfully.")


# -------------------------------------------------------------------
# Predict loop
# -------------------------------------------------------------------
while True:

    smiles = input("\nEnter SMILES (or 'q' to quit): ").strip()

    if smiles.lower() in ["q", "quit", "exit"]:
        break

    try:
        mols, final, spread = predict_NMR_C(smiles, model)

        print("\nPredicted 13C Chemical Shifts")
        print("--------------------------------")
        print(final)

        
    except Exception as e:
        print("\nPrediction failed.")
        print(e)