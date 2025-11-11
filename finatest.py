# app.py
# --- Application Streamlit : Entropy–AHP–TOPSIS ---
# Auteur : Aya Manyani 🌸

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Entropy–AHP–TOPSIS", layout="wide")

st.title("🔹 Méthode Entropy–AHP–TOPSIS")
st.write("Cette application permet d’évaluer des alternatives selon plusieurs critères, en combinant les poids issus des méthodes Entropy et AHP, puis en appliquant TOPSIS pour le classement final.")

# -------------------------------------------------
# Étape 1 : Entrée des données
# -------------------------------------------------
st.header("1️⃣ Construire la matrice de décision")

m = st.number_input("Nombre d'alternatives (m)", min_value=2, value=3, step=1)
n = st.number_input("Nombre de critères (n)", min_value=2, value=3, step=1)

alt_names = [f"A{i+1}" for i in range(m)]
crit_names = [f"C{j+1}" for j in range(n)]

st.write("### Entrez les valeurs de la matrice de décision :")
df = pd.DataFrame(np.random.rand(m, n), index=alt_names, columns=crit_names)
df_numeric = st.data_editor(df, num_rows="fixed")

st.success("✅ Matrice de décision enregistrée !")

# -------------------------------------------------
# Étape 2 : Normalisation de la matrice
# -------------------------------------------------
st.header("2️⃣ Normalisation de la matrice")
norm_matrix = df_numeric / np.sqrt((df_numeric ** 2).sum())
st.dataframe(norm_matrix.style.format("{:.4f}"))

# -------------------------------------------------
# Étape 3 : Poids par la méthode d’Entropie
# -------------------------------------------------
st.header("3️⃣ Calcul des poids par la méthode d'entropie des sous criteres")

pij = norm_matrix / norm_matrix.sum()
epsilon = 1e-12
ej = -(1 / np.log(m)) * (pij * np.log(pij + epsilon)).sum()
wej = (1 - ej) / (1 - ej).sum()

st.write("**Poids Entropie (wₑⱼ)**")
st.dataframe(pd.DataFrame(wej, columns=["Poids Entropie"]).T.style.format("{:.4f}"))

st.header("3️⃣ es poids par la méthode d'entropie des criteres A B C ")
st.write("**Poids Entropie (wₑi)**")
st.dataframe(pd.DataFrame(wei, columns=["Poids Entropie"]).T.style.format("{:.4f}"))
# -------------------------------------------------
# Étape 4 : Poids par la méthode AHP
# -------------------------------------------------
st.header("4️⃣ Poids tota par la méthode AHP ")
st.write("**Poids AHP normalisés (wₕⱼ)**")
st.dataframe(pd.DataFrame(ahp_weights, columns=["Poids AHP"]).T.style.format("{:.4f}"))

# -------------------------------------------------
# Étape 5 : Combinaison des poids Entropy et AHP
# -------------------------------------------------
st.header("5️⃣ Combinaison pondérée des poids")

combined = (wej*wei*ahp_weights) / ((wei*wej) * ahp_weights).sum()

st.write("**Poids combinés (w𝑐ⱼ)**")
st.dataframe(pd.DataFrame(combined, columns=["Poids combinés"]).T.style.format("{:.4f}"))

# -------------------------------------------------
# Étape 6 : Matrice pondérée
# -------------------------------------------------
st.header("6️⃣ Matrice de décision pondérée")

weighted_matrix = norm_matrix * combined
st.dataframe(weighted_matrix.style.format("{:.4f}"))

# -------------------------------------------------
# Étape 7 : TOPSIS - Solutions idéales et distances
# -------------------------------------------------
st.header("7️⃣ TOPSIS – Solutions idéales et distances")

benefit_criteria = st.multiselect("Critères de type bénéfice :", crit_names, default=crit_names)
cost_criteria = [c for c in crit_names if c not in benefit_criteria]

positive_ideal = np.array([weighted_matrix[c].max() if c in benefit_criteria else weighted_matrix[c].min() for c in crit_names])
negative_ideal = np.array([weighted_matrix[c].min() if c in benefit_criteria else weighted_matrix[c].max() for c in crit_names])

dist_pos = np.sqrt(((weighted_matrix - positive_ideal) ** 2).sum(axis=1))
dist_neg = np.sqrt(((weighted_matrix - negative_ideal) ** 2).sum(axis=1))

closeness = dist_neg / (dist_pos + dist_neg)

# -------------------------------------------------
# Étape 8 : Résultats finaux
# -------------------------------------------------
st.header("8️⃣ Résultats finaux – Classement des alternatives")

results = pd.DataFrame({
    "Distance + (PIS)": dist_pos,
    "Distance - (NIS)": dist_neg,
    "Closeness (Cᵢ)": closeness,
}, index=alt_names).sort_values(by="Closeness (Cᵢ)", ascending=False)

st.dataframe(results.style.format("{:.4f}"))

best_alt = results.index[0]
st.success(f"🏆 L’alternative la plus performante est **{best_alt}** avec un score de proximité de {results.iloc[0, 2]:.4f}")

st.caption("Développé par Aya Manyani 🌸 – Méthode Entropy–AHP–TOPSIS complète.")

