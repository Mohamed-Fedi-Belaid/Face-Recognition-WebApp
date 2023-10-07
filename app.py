import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage import color, transform, feature
import matplotlib.pyplot as plt


# Charger le modèle
loaded_model = joblib.load('face_detection_model.sav')

# Titre et description de l'application
st.title("Détection de Visages")
st.markdown("Téléchargez une image et cliquez sur le bouton pour détecter les visages.")

# Sélectionner une image depuis l'ordinateur
uploaded_image = st.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])

def sliding_window(img, patch_size=(62, 47), istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

def draw_red_rectangles(image, indices, labels):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    Ni, Nj = (42,67)
    indices = np.array(indices)

    for i, j in indices[labels == 1]:
        ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))

    return fig

if uploaded_image is not None:
    # Charger l'image et la prétraiter
    img = Image.open(uploaded_image)
    img = np.array(img)
    gray_img = color.rgb2gray(img)
    resized_img = transform.rescale(gray_img, 0.5)
    cropped_img = resized_img

    # Afficher l'image
    st.image(cropped_img, caption="Image Téléchargée", use_column_width=True)

    # Bouton pour détecter les visages
    if st.button("Détecter les visages"):
        # Extraire les patches
        indices, patches = zip(*sliding_window(cropped_img))
        patches_hog = np.array([feature.hog(patch) for patch in patches])

        # Prédire les visages
        labels = loaded_model.predict(patches_hog)

        # Dessiner les rectangles rouges autour des visages détectés
        fig = draw_red_rectangles(cropped_img, indices, labels)
        st.pyplot(fig)
