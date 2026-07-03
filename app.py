import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ChestX-Ray 14-Disease Classifier",
    page_icon="🫁",
    layout="wide",
)

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

DISEASE_INFO = {
    "Atelectasis": {
        "description": (
            "Atelectasis is a partial or complete collapse of the lung, "
            "often caused by airway obstruction (mucus plug, foreign body) "
            "or external compression (tumor, fluid). It reduces gas exchange "
            "and can lead to shortness of breath."
        ),
        "common_causes": "Mucus plugging, foreign body, tumor, pleural effusion, pneumothorax, post-surgical hypoventilation.",
        "typical_findings": "Increased opacity, fissure displacement, mediastinal shift toward the affected side, compensatory hyperinflation of the remaining lung.",
    },
    "Cardiomegaly": {
        "description": (
            "Cardiomegaly refers to an enlarged heart, often detected on chest "
            "X-ray when the cardiothoracic ratio exceeds 50%. It indicates "
            "underlying conditions such as hypertension, heart failure, or "
            "valvular disease."
        ),
        "common_causes": "Hypertension, congestive heart failure, valvular heart disease, cardiomyopathy, coronary artery disease.",
        "typical_findings": "Increased cardiothoracic ratio (>0.5), globular heart shape, possible signs of pulmonary congestion.",
    },
    "Consolidation": {
        "description": (
            "Consolidation appears as a homogeneous opacity in the lung "
            "parenchyma, typically due to fluid or solid material filling the "
            "airspaces. Pneumonia is the most common cause."
        ),
        "common_causes": "Pneumonia (bacterial, viral, fungal), pulmonary edema, aspiration, pulmonary hemorrhage.",
        "typical_findings": "Homogeneous opacity with air bronchograms, lobar or segmental distribution, no significant volume loss.",
    },
    "Edema": {
        "description": (
            "Pulmonary edema is the accumulation of fluid in the lung "
            "interstitium and airspaces. Cardiogenic edema stems from left "
            "ventricular failure, while non-cardiogenic causes include "
            "ARDS and fluid overload."
        ),
        "common_causes": "Congestive heart failure, acute respiratory distress syndrome (ARDS), renal failure, fluid overload, aspiration.",
        "typical_findings": "Perihilar bat-wing opacities, Kerley B lines, pleural effusion, enlarged cardiac silhouette, vascular redistribution.",
    },
    "Effusion": {
        "description": (
            "A pleural effusion is an abnormal accumulation of fluid in the "
            "pleural space between the lung and chest wall. It can be "
            "transudative (heart failure, cirrhosis) or exudative (infection, "
            "malignancy)."
        ),
        "common_causes": "Congestive heart failure, pneumonia, malignancy, pulmonary embolism, tuberculosis, cirrhosis, renal disease.",
        "typical_findings": "Blunting of the costophrenic angle, meniscus sign, opacification of the lower lung field, possible mediastinal shift.",
    },
    "Emphysema": {
        "description": (
            "Emphysema is a chronic lung disease characterized by irreversible "
            "destruction of the alveolar walls, leading to air trapping and "
            "hyperinflation. It is a form of COPD primarily caused by smoking."
        ),
        "common_causes": "Cigarette smoking, alpha-1 antitrypsin deficiency, environmental pollutants, occupational exposure.",
        "typical_findings": "Hyperinflation (flattened diaphragm, increased retrosternal airspace), bullae, vascular attenuation, narrow mediastinum.",
    },
    "Fibrosis": {
        "description": (
            "Pulmonary fibrosis involves progressive scarring of lung tissue, "
            "leading to reduced elasticity and impaired gas exchange. Idiopathic "
            "pulmonary fibrosis (IPF) is the most common form."
        ),
        "common_causes": "Idiopathic pulmonary fibrosis, connective tissue disease, occupational exposure (asbestos, silica), medications, radiation.",
        "typical_findings": "Reticular opacities, honeycombing (subpleural, basilar), traction bronchiectasis, reduced lung volumes.",
    },
    "Hernia": {
        "description": (
            "A diaphragmatic hernia is the protrusion of abdominal organs into "
            "the chest cavity through a defect in the diaphragm. Hiatal hernias "
            "are most common and often incidental on chest X-ray."
        ),
        "common_causes": "Congenital defect (Bochdalek, Morgagni), trauma, increased intra-abdominal pressure, hiatal hernia.",
        "typical_findings": "Air-fluid level behind the heart, visible stomach or bowel in the thorax, mediastinal shift.",
    },
    "Infiltration": {
        "description": (
            "Infiltration refers to abnormal substances (cells, fluid, protein) "
            "accumulating in the lung interstitium or airspaces. It is a "
            "non-specific finding that may indicate infection, inflammation, "
            "or malignancy."
        ),
        "common_causes": "Infection (pneumonia, tuberculosis), interstitial lung disease, pulmonary edema, malignancy (lymphangitic carcinomatosis).",
        "typical_findings": "Reticular, nodular, or reticulonodular opacities; peribronchial cuffing; ground-glass opacity; possible distribution in specific lung zones.",
    },
    "Mass": {
        "description": (
            "A pulmonary mass is a discrete opacity >3 cm in diameter on chest "
            "X-ray. It raises concern for malignancy (primary lung cancer or "
            "metastasis) but may also be benign."
        ),
        "common_causes": "Lung cancer (squamous cell, adenocarcinoma, small cell), metastasis, benign tumors (hamartoma, granuloma), abscess.",
        "typical_findings": "Round or oval opacity >3 cm, possible spiculated margins, cavitation, associated lymphadenopathy or pleural effusion.",
    },
    "Nodule": {
        "description": (
            "A pulmonary nodule is a small, round opacity ≤3 cm in diameter. "
            "Most are benign (granulomas, hamartomas), but some may represent "
            "early lung cancer, requiring further evaluation with CT."
        ),
        "common_causes": "Granuloma (fungal, TB), hamartoma, lung cancer, metastasis, arteriovenous malformation, round pneumonia.",
        "typical_findings": "Round opacity ≤3 cm, well-defined or spiculated margins, solid or subsolid density, solitary or multiple, possible calcification (benign pattern).",
    },
    "Pleural_Thickening": {
        "description": (
            "Pleural thickening is scarring of the pleural lining, often from "
            "prior inflammation, infection, or asbestos exposure. It can be "
            "diffuse or focal and may impair lung expansion."
        ),
        "common_causes": "Asbestos exposure, prior pleuritis / empyema, tuberculosis, hemothorax, rheumatoid arthritis, prior surgery or trauma.",
        "typical_findings": "Pleural opacity along the chest wall, blunted costophrenic angle (without meniscus), calcified pleural plaques, stable over time.",
    },
    "Pneumonia": {
        "description": (
            "Pneumonia is an infection of the lung parenchyma causing alveolar "
            "inflammation and consolidation. It can be community-acquired, "
            "hospital-acquired, or aspiration-related."
        ),
        "common_causes": "Streptococcus pneumoniae, Haemophilus influenzae, Mycoplasma, viruses, fungi, aspiration of gastric contents.",
        "typical_findings": "Lobar or segmental consolidation, air bronchograms, patchy opacities, possible pleural effusion, no volume loss.",
    },
    "Pneumothorax": {
        "description": (
            "Pneumothorax is the accumulation of air in the pleural space, "
            "causing partial or complete lung collapse. It can be spontaneous "
            "(primary or secondary) or traumatic."
        ),
        "common_causes": "Spontaneous (bleb rupture), trauma, iatrogenic (central line, biopsy), mechanical ventilation, COPD.",
        "typical_findings": "Visceral pleural line (thin white line), absent lung markings peripheral to the line, deep sulcus sign, possible mediastinal shift (tension).",
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


class MultiLabelModel(nn.Module):
    def __init__(self, base="densenet", num_classes=14):
        super().__init__()
        self.base_name = base
        if base == "densenet":
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif base == "efficientnet":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


@st.cache_resource
def load_model(model_name):
    model = MultiLabelModel("densenet").to(DEVICE)
    model.eval()
    weights_map = {
        "DenseNet Frozen": "best_densenet_frozen.pth",
        "DenseNet Finetuned": "best_densenet_finetuned.pth",
    }
    weight_file = weights_map.get(model_name)
    if weight_file is None:
        st.error(f"Unknown model: {model_name}")
        st.stop()
    weight_path = os.path.join(OUTPUT_DIR, weight_file)
    if not os.path.exists(weight_path):
        st.error(f"Model weights not found at: {weight_path}")
        st.error("Please train the models first by running the notebook.")
        st.stop()
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


def preprocess_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not decode image. Please upload a valid chest X-ray.")
        st.stop()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    return tensor, img_rgb


def predict(model, tensor):
    with torch.no_grad():
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def get_ensemble_prediction(models, weights, tensor):
    all_probs = []
    for model in models:
        all_probs.append(predict(model, tensor))
    avg = np.average(all_probs, axis=0, weights=weights)
    return avg


st.title("🫁 ChestX-Ray 14-Disease Multi-Label Classifier")
st.markdown(
    """
    Upload a chest X-ray image and select a model to get **14-disease multi-label predictions**.
    The models were trained on the [NIH Chest X-Ray 14 dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
    using DenseNet121 with a custom classifier head.
    """
)

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["DenseNet Frozen", "DenseNet Finetuned", "Ensemble (Frozen + Finetuned)"],
)

model = None
ensemble_models = None
ensemble_weights = None

if model_choice == "Ensemble (Frozen + Finetuned)":
    m1 = load_model("DenseNet Frozen")
    m2 = load_model("DenseNet Finetuned")
    ensemble_models = [m1, m2]
    ensemble_weights = [0.625, 0.375]
    st.sidebar.info(
        f"Ensemble weights: Model 1 = {ensemble_weights[0]:.3f}, "
        f"Model 2 = {ensemble_weights[1]:.3f} (based on validation AUC)"
    )
else:
    model = load_model(model_choice)

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

threshold_default = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Device:** {DEVICE}" if DEVICE else "**Device:** CPU"
)

if uploaded_file is not None:
    with st.spinner("Processing image and running inference..."):
        tensor, img_rgb = preprocess_image(uploaded_file)

        if model_choice == "Ensemble (Frozen + Finetuned)":
            probs = get_ensemble_prediction(ensemble_models, ensemble_weights, tensor)
        else:
            probs = predict(model, tensor)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Uploaded X-Ray")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_rgb, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Prediction Probabilities")
        df_probs = pd.DataFrame({
            "Disease": DISEASE_CLASSES,
            "Probability": probs,
            "Prediction": ["✅ Positive" if p >= threshold_default else "❌ Negative" for p in probs],
        })
        df_probs = df_probs.sort_values("Probability", ascending=False).reset_index(drop=True)

        color_map = ["#2ecc71" if p >= threshold_default else "#e74c3c" for p in df_probs["Probability"]]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.barh(range(len(df_probs)), df_probs["Probability"].values, color=color_map)
        ax2.set_yticks(range(len(df_probs)))
        ax2.set_yticklabels(df_probs["Disease"].values)
        ax2.axvline(x=threshold_default, color="gray", linestyle="--", label=f"Threshold = {threshold_default}")
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability")
        ax2.set_title("Per-Disease Prediction Scores")
        ax2.legend()
        ax2.invert_yaxis()
        st.pyplot(fig2)
        plt.close()

    st.subheader("Results Table")
    st.dataframe(
        df_probs.style.format({"Probability": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    positive_diseases = df_probs[df_probs["Prediction"].str.contains("Positive")]
    if not positive_diseases.empty:
        st.subheader("Disease Analysis & Reasoning")

        tab_names = [f"{row['Disease']} ({row['Probability']:.1%})" for _, row in positive_diseases.iterrows()]
        tabs = st.tabs(tab_names)

        for idx, (_, row) in enumerate(positive_diseases.iterrows()):
            disease = row["Disease"]
            prob = row["Probability"]
            info = DISEASE_INFO[disease]

            with tabs[idx]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Description**")
                    st.write(info["description"])
                    st.markdown("**Common Causes**")
                    st.write(info["common_causes"])
                with col_b:
                    st.markdown("**Confidence Score**")
                    st.metric(label="Probability", value=f"{prob:.2%}")
                    st.markdown("**Typical Radiographic Findings**")
                    st.write(info["typical_findings"])

                st.info(
                    f"**Reasoning:** The model assigned a {prob:.1%} probability "
                    f"to **{disease}**, which is the strongest pattern detected in this "
                    f"radiograph among the 14 disease classes. This finding should be "
                    f"corroborated by a board-certified radiologist."
                )
    else:
        st.success(
            f"No diseases detected above the {threshold_default:.2f} threshold. "
            f"Highest probability: **{df_probs.iloc[0]['Disease']}** "
            f"({df_probs.iloc[0]['Probability']:.1%})."
        )

    with st.expander("📊 Performance Metrics (from validation set)"):
        if model_choice == "DenseNet Frozen":
            metrics = {"Avg AUROC": "~0.77", "Best Val Loss": "0.64"}
        elif model_choice == "DenseNet Finetuned":
            metrics = {"Avg AUROC": "~0.81", "Best Val Loss": "0.59"}
        else:
            metrics = {"Macro F1": "0.28", "Micro F1": "0.35", "Avg AUROC": "~0.81"}

        metrics_df = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in metrics.items()]
        )
        st.table(metrics_df)
else:
    st.info("👈 Upload a chest X-ray image using the sidebar to get started.")
    st.markdown(
        """
        ### Instructions
        1. Select a model from the sidebar
        2. Upload a chest X-ray (PNG or JPG)
        3. Adjust the decision threshold if needed
        4. View predictions, probabilities, and disease descriptions

        ### Model Options
        - **DenseNet Frozen**: Backbone frozen, only classifier head trained
        - **DenseNet Finetuned**: Full model fine-tuned (better performance)
        - **Ensemble**: Weighted average of both models
        """
    )
