import streamlit as st
import joblib
import numpy as np
import cv2
from io import BytesIO

# Chargement du modèle sauvegardé
model = joblib.load('modelLab1.sav')

# Titre de l'application
st.title("FACE DETECTION APPLICATION")

# Section de téléchargement de fichier
st.header("Téléchargez une image pour la détection de visage")
uploaded_image = st.file_uploader("Sélectionnez une image...", type=["jpg", "png", "jpeg"])

# Vérifie si un fichier a été téléchargé
if uploaded_image is not None:
    # Affiche l'image téléchargée
    st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)

# Fonction pour prétraiter l'image et effectuer la classification
def classify_image(image):
    # Enregistrez l'image téléchargée temporairement sur le disque
    with BytesIO() as buffer:
        buffer.write(image.read())
        buffer.seek(0)
        img = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), -1)
    
    # Redimensionnez l'image et convertissez-la en niveaux de gris
    img = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray / 255.0  # Normalisation des pixels
    
    # Aplatir l'image en un vecteur 1D
    img_flat = img_gray.flatten()

     # Vérifiez que la longueur du vecteur est correcte (1215)
    if len(img_flat) != 1215:
        st.error("La longueur du vecteur d'images n'est pas correcte.")
        return None

    # Effectuer la classification avec le modèle
    result = model.predict([img_flat])  # Notez l'utilisation de crochets pour créer une liste de 1 échantillon
    return result[0] if result is not None else None

if st.button("FACE DETECTION"):
    # Classification de l'image
    result = classify_image(uploaded_image)

    # Afficher le résultat
    if result is not None:
        if result > 0.5:
            st.success("Cette image contient un visage.")
        else:
            st.warning("Cette image ne contient pas de visage.")
    else:
        st.error("Erreur lors de la classification de l'image.")

# Note d'information
st.info("Cette application permet de télécharger une image et de détecter les visages dans l'image.")

# Note de pied de page
st.text("Réalisé par Mohamed Fedi BELAID")
