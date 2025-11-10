# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Entropy-AHP-weighted TOPSIS", layout="wide")

st.title("Entropy → AHP → Weighted TOPSIS")
st.markdown("Upload / paste your decision matrix (alternatives × criteria). Then choose benefit/cost and optionally provide an AHP pairwise matrix.")

# ------------------ Helpers ------------------
RI_VALUES = {1:0.0,2:0.0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}

def parse_matrix_text(text):
    """Parse pasted matrix text (commas/spaces) into numpy array."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        ln = ln.replace(",", " ").replace(";", " ")
        parts = [p for p in ln.split() if p]
        row = []
        for p in parts:
            try:
                # allow fractions like 1/3
                if "/" in p:
                    num, den = p.split("/")
                    row.append(float(num)/float(den))
                else:
                    row.append(float(p))
            except:
                raise ValueError(f"Can't parse value: '{p}'")
        rows.append(row)
    return np.array(rows, dtype=float)

def ahp_weights_from_matrix(M):
    """Compute AHP weights via principal eigenvector and CR."""
    eigvals, eigvecs = np.linalg.eig(M)
    max_idx = np.argmax(eigvals.real)
    lam = eigvals.real[max_idx]
    vec = eigvecs[:, max_idx].real
    vec = np.maximum(vec, 0)  # numerical safety
    w = vec / np.sum(vec)
    n = M.shape[0]
    CI = (lam - n) / (n - 1) if n>1 else 0
    RI = RI_VALUES.get(n, 1.49)
    CR = CI/RI if RI != 0 else 0
    return w, lam, CI, CR

def entropy_weights(X):
    """Compute entropy weights column-wise.
       X: m x n decision matrix (non-negative)."""
    X = np.array(X, dtype=float)
    m, n = X.shape
    # avoid zero columns
    col_sum = X.sum(axis=0)
    if np.any(col_sum == 0):
        raise ValueError("At least one criterion column sums to 0; can't normalize.")
    P = X / col_sum  # p_ij = x_ij / sum_i x_ij
    # replace zeros to avoid log(0)
    eps = 1e-12
    P_safe = np.where(P <= 0, eps, P)
    k = 1.0 / np.log(m)
    e = -k * np.sum(P_safe * np.log(P_safe), axis=0)  # entropy per criterion
    d = 1 - e
    if d.sum() == 0:
        w = np.ones(n) / n
    else:
        w = d / np.sum(d)
    return w, P

# ------------------ UI: Input ------------------
st.header("1) Input matrix (alternatives × criteria)")

col1, col2 = st.columns([2,1])
with col1:
    upload = st.file_uploader("Upload CSV or Excel (alternatives rows, criteria cols)", type=["csv","xlsx"])
    paste = st.text_area("Or paste matrix text (rows on new lines, values separated by space/comma).", height=160)

with col2:
    st.write("Examples:")
    st.write("- CSV with header row = criteria names, index (optional) = alternatives names")
    st.write("- Paste:\n1 2 3\n4 5 6\n7 8 9")
    st.write("You may use fractions like 1/3.")

# Try to read matrix
df = None
alt_names = None
crit_names = None
if upload is not None:
    try:
        if upload.name.endswith(".csv"):
            df = pd.read_csv(upload, index_col=0)
        else:
            df = pd.read_excel(upload, index_col=0)
        if df.shape[0] < 1 or df.shape[1] < 1:
            st.error("Uploaded file doesn't look like a valid decision matrix with rows and columns.")
            df = None
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        df = None

if df is None and paste.strip() != "":
    try:
        arr = parse_matrix_text(paste)
        df = pd.DataFrame(arr)
    except Exception as e:
        st.error(f"Can't parse pasted matrix: {e}")
        df = None

if df is None:
    st.info("Please upload a matrix file or paste matrix values to continue.")
    st.stop()

# If df exists:
st.success("Decision matrix loaded.")
# let user adjust names
with st.expander("View / edit matrix"):
    if df.index.dtype == object:
        alt_names = list(df.index.astype(str))
    else:
        alt_names = [f"A{i+1}" for i in range(df.shape[0])]
    if df.columns.dtype == object:
        crit_names = list(df.columns.astype(str))
    else:
        crit_names = [f"C{j+1}" for j in range(df.shape[1])]

    # allow editing names
    st.write("Alternatives (rows):")
    for i in range(len(alt_names)):
        alt_names[i] = st.text_input(f"Alt name {i+1}", value=alt_names[i], key=f"alt{i}")

    st.write("Criteria (cols):")
    for j in range(len(crit_names)):
        crit_names[j] = st.text_input(f"Crit name {j+1}", value=crit_names[j], key=f"crit{j}")

    # show numeric matrix editable
    try:
        df_numeric = df.astype(float)
    except:
        # attempt to parse object to float
        df_numeric = df.applymap(lambda x: float(str(x).replace(",",".").strip()))
    edited = st.experimental_data_editor(df_numeric, num_rows="fixed")
    df = edited
    df.index = alt_names
    df.columns = crit_names

# ------------------ UI: criteria type and AHP matrix ------------------
st.header("2) Criteria type & AHP weights (optional)")

cols = st.columns(len(crit_names))
crit_types = []
for j, c in enumerate(crit_names):
    with cols[j]:
        ct = st.selectbox(f"{c}", options=["benefit","cost"], key=f"type_{j}")
        crit_types.append(ct)

st.markdown("**AHP weights**: upload a pairwise comparison matrix (n×n), paste it, or use equal weights.")
ahp_upload = st.file_uploader("Upload AHP pairwise matrix (CSV/Excel) (criteria × criteria)", type=["csv","xlsx"], key="ahp_upload")
ahp_paste = st.text_area("Or paste AHP matrix text (rows newlines). Leave empty to skip.", key="ahp_paste", height=120)
ahp_matrix = None
if ahp_upload is not None:
    try:
        if ahp_upload.name.endswith(".csv"):
            mahp = pd.read_csv(ahp_upload, index_col=0)
        else:
            mahp = pd.read_excel(ahp_upload, index_col=0)
        ahp_matrix = mahp.values.astype(float)
    except Exception as e:
        st.error(f"Can't read AHP upload: {e}")
        ahp_matrix = None

if ahp_matrix is None and ahp_paste.strip() != "":
    try:
        ahp_matrix = parse_matrix_text(ahp_paste)
    except Exception as e:
        st.error(f"Can't parse AHP paste: {e}")
        ahp_matrix = None

use_equal_ahp = False
if ahp_matrix is None:
    use_equal_ahp = st.checkbox("Use equal AHP weights (skip AHP matrix)", value=True)

# alpha: how to combine? We'll follow provided formula: multiply and normalize.
st.markdown("**Note on combination**: objective (entropy) and subjective (AHP) weights are multiplied and normalized (wc_j ∝ w_entropy_j * w_AHP_j).")

# ------------------ Calculations ------------------
X = df.values.astype(float)  # m x n
m, n = X.shape

# Step2: Normalize decision matrix (column-wise)
try:
    ent_w, P = entropy_weights(X)  # ent_w length n, P is normalized
except Exception as e:
    st.error(f"Entropy normalization error: {e}")
    st.stop()

# Step3: entropy weights already computed (ent_w)

# Step4: AHP weights
if not use_equal_ahp and ahp_matrix is not None:
    if ahp_matrix.shape != (n, n):
        st.error(f"AHP matrix must be {n}x{n}. Provided: {ahp_matrix.shape}")
        st.stop()
    try:
        ahp_w, lam, CI, CR = ahp_weights_from_matrix(np.array(ahp_matrix, dtype=float))
    except Exception as e:
        st.error(f"AHP computation error: {e}")
        st.stop()
else:
    ahp_w = np.ones(n) / n
    lam = None
    CI = None
    CR = None

# Step5: combination weight (product then normalized)
prod = ent_w * ahp_w
if prod.sum() == 0:
    comb_w = np.ones(n) / n
else:
    comb_w = prod / prod.sum()

# Step6: entropy-AHP weight matrix: in original text there was wcj = 1wci 2wcj? We'll stick to vector comb_w as criterion weights.

# Step7: build weighted normalized decision matrix uij = wcj * pij
U = P * comb_w[np.newaxis, :]

# Step8: positive/negative ideals:
U_plus = np.empty(n)
U_minus = np.empty(n)
for j in range(n):
    if crit_types[j] == "benefit":
        U_plus[j] = U[:,j].max()
        U_minus[j] = U[:,j].min()
    else:  # cost
        U_plus[j] = U[:,j].min()
        U_minus[j] = U[:,j].max()

# Step9: distances
S_plus = np.sqrt(np.sum((U - U_plus)**2, axis=1))
S_minus = np.sqrt(np.sum((U - U_minus)**2, axis=1))

# Step10: relative closeness
with np.errstate(divide='ignore', invalid='ignore'):
    C = S_minus / (S_plus + S_minus)
# Step11: ranking
rank_idx = np.argsort(-C)  # descending

# ------------------ Output ------------------
st.header("Results")

colA, colB = st.columns([2,1])
with colA:
    st.subheader("Weighted normalized decision matrix (U)")
    dfU = pd.DataFrame(U, index=alt_names, columns=crit_names).round(6)
    st.dataframe(dfU)

    st.subheader("Alternatives ranking (by relative closeness to PIS)")
    result_df = pd.DataFrame({
        "Alternative": alt_names,
        "S_plus": S_plus.round(6),
        "S_minus": S_minus.round(6),
        "Closeness": C.round(6)
    })
    result_df["Rank"] = result_df["Closeness"].rank(method="dense", ascending=False).astype(int)
    result_df = result_df.sort_values(by="Closeness", ascending=False).reset_index(drop=True)
    st.dataframe(result_df)

with colB:
    st.subheader("Weights")
    w_df = pd.DataFrame({
        "Criteria": crit_names,
        "Entropy_w": ent_w.round(6),
        "AHP_w": ahp_w.round(6),
        "Combined_w": comb_w.round(6),
        "Type": crit_types
    })
    st.table(w_df)

    if lam is not None:
        st.markdown(f"**AHP λmax:** {lam:.4f}")
        st.markdown(f"**AHP CI:** {CI:.4f}")
        st.markdown(f"**AHP CR:** {CR:.4f}")
        if CR >= 0.1:
            st.warning("AHP consistency ratio CR >= 0.1 → pairwise comparisons may be inconsistent.")

# Download results
st.markdown("---")
st.subheader("Download results")
buf = io.StringIO()
result_df.to_csv(buf, index=False)
btn = st.download_button("Download ranking CSV", data=buf.getvalue(), file_name="topsis_ranking.csv", mime="text/csv")

# Show PIS/NIS
with st.expander("Show Positive (PIS) and Negative (NIS) ideals"):
    st.write("PIS (U+):", pd.Series(U_plus, index=crit_names).round(6))
    st.write("NIS (U-):", pd.Series(U_minus, index=crit_names).round(6))

st.markdown("Done — method: Entropy (objective) + AHP (subjective) combined by product then normalized → Weighted TOPSIS.")
st.caption("Developed by Aya — modify as needed.")
