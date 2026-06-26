import os, re, ast, pickle, unicodedata, hashlib
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse
import streamlit as st
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — ajusta las rutas según tu estructura de carpetas
# ─────────────────────────────────────────────────────────────────────────────
RUTA_MALLAS   = "tokenizado_cursos.csv"
RUTA_OFERTAS  = "df_ofertas_1.csv"
RUTA_ONLINE   = "palabras_3.csv"
RUTA_COURSERA = "coursera_cursos_v5.csv"
RUTA_UDEMY    = "udemy_cursos_playwright_3.csv"
RUTA_W2V      = "modelos/word2vec.model"
RUTA_TFIDF    = "modelos/tfidf.pkl"
RUTA_TFIDF_X  = "modelos/tfidf_X.npz"
os.makedirs("modelos", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE TEXTO
# ─────────────────────────────────────────────────────────────────────────────
def quitar_acentos(texto):
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def parse_lista(valor):
    if isinstance(valor, list):
        return valor
    if not isinstance(valor, str) or not valor.strip():
        return []
    try:
        out = ast.literal_eval(valor)
        return out if isinstance(out, list) else [str(out)]
    except (ValueError, SyntaxError):
        return re.findall(r"[a-zA-Z0-9_]+", valor)

def limpiar_tokens(lista, stop=None, min_len=2, dedup=False):
    stop = stop or set()
    resultado = [t for t in lista if len(t) >= min_len and t not in stop]
    if dedup:
        seen, out = set(), []
        for t in resultado:
            if t not in seen:
                seen.add(t); out.append(t)
        return out
    return resultado

STOP_CURSO = {
    "fundamentos","introduccion","electivo","electivos","taller","seminario",
    "laboratorio","trabajo","practica","practicas","proyecto","proyectos",
    "aplicaciones","ingles","comunicacion","humanidades","general","basico",
    "nivel","parte","modulo","area","aspectos","ingenieria","ciencia","ciencias",
}
STOP_OFERTAS = {
    "intermedio","avanzado","basico","nivel","conocimiento","conocimientos",
    "experiencia","manejo","herramienta","herramientas","requisito","requisitos",
    "deseable","indispensable","capacidad","habilidad","habilidades","competencia",
    "competencias","minimo","anos","ano","roles","posiciones","perfil","puesto",
    "soporte","fuerte","enfoque","manera","presencial","remoto","horas","realizando",
    "estudios","encontramos","orientado","generacion","idealmente","evaluara",
    "proceso","asistente","analista","formar","estudiante","sector",
}
STOP_CURSOS_ONLINE = {
    "aprender","aprenderas","aprendera","crear","poder","tener","utilizar","usar",
    "permitir","realizar","desarrollar","disenar","trabajar","comenzar","empezar",
    "conocer","entender","comprender","dominar","gestionar","manejar","optimizar",
    "mejorar","generar","obtener","construir","implementar","analizar","aplicar",
    "llevar","dar","ver","saber","querer","podras","incluir","pasar","buscar",
    "ensenar","ayudar","tomar","disenado","finalizar",
    "curso","clase","leccion","modulo","seccion","video","material","recurso",
    "ejercicio","tarea","ejemplo","tema","parte","paso","fundamento","concepto",
    "nivel","basico","avanzado","practico","completo","introductorio","introductoria",
    "introduccion","programa","especializado","formacion","aprendizaje","conocimiento",
    "habilidad","contenido","objetivo",
    "facil","rapido","sencillo","claro","efectivo","diferente","alguno","mismo",
    "propio","nuevo","importante","necesario","principal","dinamico","largo","previo",
    "clave","hora","dia","vez","tiempo","primero","segundo","final","cero","mundo",
    "vida","forma","manera",
    "and","the","for","with","from","you","your",
    "estudiante","alumno","persona","profesional","experto","usuario","cualquiera",
    "cliente","online","acceso","ingles","traves","ademas","mediante","dentro","creer",
}
TOKENS_NO_TECH = {
    "calculo","algebra","fisica","quimica","matematica","matematicas","geometria",
    "trigonometria","vectores","mecanica","termodinamica","economia","contabilidad",
    "administracion","finanzas","contable","geografia","historia","filosofia",
    "sociologia","politica","derecho","etica","religion","teologia","psicologia",
    "antropologia","redaccion","escritura","oratoria","lenguaje","literatura",
    "periodismo","linguistica","marketing","publicidad","ventas","emprendimiento",
    "liderazgo","negociacion","logistica","tesis","ambiental","sostenibilidad",
    "ecologia","responsabilidad","salud","medicina","enfermeria","nutricion",
    "nivelacion","deportes","arte","musica","teatro",
}
SEMILLAS_TECH   = [
    "python","sql","datos","algoritmos","software","redes","seguridad","cloud",
    "programacion","sistemas","machine_learning","base_datos","desarrollo","automatizacion",
]
UMBRAL_SIM_TECH = 0.30
PATRON_IDIOMAS  = re.compile(
    r"\b(ingl[eé]s|english|franc[eé]s|french|alem[aá]n|german|italiano|"
    r"portugu[eé]s|mandar[ií]n|chino|japon[eé]s|idioma|toefl|ielts|gram[aá]tica)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# VECTORES
# ─────────────────────────────────────────────────────────────────────────────
def vec(tokens, modelo):
    vs = [modelo.wv[t] for t in tokens if t in modelo.wv]
    return np.mean(vs, axis=0) if vs else None

def cos(a, b):
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# CARGA (cacheada)
# ─────────────────────────────────────────────────────────────────────────────
def parsear_precio(valor):
    if not isinstance(valor, str) or not valor.strip():
        return "No especificado"
    if "gratis" in valor.lower():
        return "Gratis"
    m = re.search(r"([0-9.]+,[0-9]+|[0-9]+[.,]?[0-9]*)",
                  valor.replace("Precio actual", ""))
    if m:
        try:
            return f"S/ {float(m.group(1).replace('.','').replace(',','.')):.2f}"
        except ValueError:
            pass
    return valor.strip()

@st.cache_resource(show_spinner="Cargando datos y modelos… (solo ocurre una vez)")
def cargar_todo():
    mallas = pd.read_csv(RUTA_MALLAS).rename(columns={"Unnamed: 0": "id"})
    mallas["tokens_nombre"] = mallas["tokens"].apply(
        lambda v: limpiar_tokens(parse_lista(v), stop=STOP_CURSO)
    )
    mallas["es_no_tech"] = mallas["tokens_nombre"].apply(
        lambda t: bool(set(t) & TOKENS_NO_TECH)
    )

    ofertas = pd.read_csv(RUTA_OFERTAS)
    ofertas["hab_tokens"] = ofertas["habilidades"].apply(parse_lista)
    ofertas["hab_tokens"] = ofertas["hab_tokens"].apply(
        lambda l: limpiar_tokens(l, stop=STOP_OFERTAS)
    )
    ofertas = ofertas[ofertas["hab_tokens"].map(len) > 0].reset_index(drop=True)

    online = pd.read_csv(RUTA_ONLINE, sep="|").rename(
        columns={"texto_consolidado": "descripcion", "fuente": "portal"}
    )
    online["tokens"] = online["tokens_limpios"].apply(
        lambda v: limpiar_tokens(parse_lista(v), stop=STOP_CURSOS_ONLINE, dedup=True)
    )
    online["doc"] = online["tokens"].apply(lambda t: " ".join(t))
    cou  = pd.read_csv(RUTA_COURSERA, sep=";")[["titulo","precio","url"]]
    ude  = pd.read_csv(RUTA_UDEMY,    sep=";")[["titulo","precio","url"]]
    meta = pd.concat([cou, ude], ignore_index=True).drop_duplicates("titulo", keep="first")
    meta["precio"] = meta["precio"].apply(parsear_precio)
    online = online.merge(meta, on="titulo", how="left")
    online["precio"] = online["precio"].fillna("No especificado")
    online["url"]    = online["url"].fillna("")
    online = online[online["tokens"].map(len) > 0]
    online = online[~online["titulo"].astype(str).apply(
        lambda t: bool(PATRON_IDIOMAS.search(t)))].reset_index(drop=True)

    # Word2Vec
    if os.path.exists(RUTA_W2V):
        modelo = Word2Vec.load(RUTA_W2V)
    else:
        corpus  = list(online["tokens"]) + list(ofertas["hab_tokens"]) + list(mallas["tokens_nombre"])
        corpus  = [d for d in corpus if d]
        modelo  = Word2Vec(sentences=corpus, vector_size=100, window=5,
                           min_count=2, sg=1, workers=4, epochs=20, seed=42)
        modelo.save(RUTA_W2V)

    # Filtro tech
    sem_ok = [s for s in SEMILLAS_TECH if s in modelo.wv]
    v_tech = np.mean([modelo.wv[s] for s in sem_ok], axis=0)
    cand_f1 = mallas[~mallas["es_no_tech"] & (mallas["tokens_nombre"].map(len) > 0)].copy()
    cand_f1["sim_tech"] = cand_f1["tokens_nombre"].apply(lambda t: cos(vec(t, modelo), v_tech))
    cursos_tech = set(cand_f1[cand_f1["sim_tech"] >= UMBRAL_SIM_TECH]["curso"])
    mallas["es_tech"] = mallas["curso"].isin(cursos_tech)

    freq_hab = Counter()
    for toks in ofertas["hab_tokens"]:
        freq_hab.update(set(toks))
    habilidades_validas = [h for h, n in freq_hab.items() if n >= 5 and h in modelo.wv]

    def predecir_skills(tokens_curso):
        v_curso = vec(tokens_curso, modelo)
        if v_curso is None:
            return []
        scores = [(h, cos(v_curso, modelo.wv[h])) for h in habilidades_validas]
        scores = [(h, s) for h, s in scores if s >= 0.30]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in scores[:6]]

    mallas["skills_predichos"] = mallas["tokens_nombre"].apply(predecir_skills)

    # TF-IDF
    if os.path.exists(RUTA_TFIDF) and os.path.exists(RUTA_TFIDF_X):
        with open(RUTA_TFIDF, "rb") as f:
            tfidf = pickle.load(f)
        X = scipy.sparse.load_npz(RUTA_TFIDF_X)
    else:
        tfidf = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2)
        X = tfidf.fit_transform(online["doc"])
        with open(RUTA_TFIDF, "wb") as f:
            pickle.dump(tfidf, f)
        scipy.sparse.save_npz(RUTA_TFIDF_X, X)

    return mallas, ofertas, online, modelo, habilidades_validas, tfidf, X

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def filtrar_ofertas_por_carrera(ofertas, carrera, modelo, top_pct=0.5):
    carrera_tokens = limpiar_tokens(quitar_acentos(carrera).lower().split())
    v_carrera = vec(carrera_tokens, modelo)
    sims = [cos(vec(toks, modelo), v_carrera) for toks in ofertas["hab_tokens"]]
    ofertas = ofertas.copy()
    ofertas["sim_carrera"] = sims
    umbral = np.quantile(sims, 1 - top_pct)
    return ofertas[ofertas["sim_carrera"] >= umbral].reset_index(drop=True)

def construir_demanda(ofertas_rel):
    cnt = Counter()
    for toks in ofertas_rel["hab_tokens"]:
        cnt.update(set(toks))
    return dict(cnt)

def calcular_gap(demanda, tokens_cubiertos, modelo, umbral=0.65):
    vecs_cub = {t: modelo.wv[t] for t in tokens_cubiertos if t in modelo.wv}
    filas = []
    for skill, freq in demanda.items():
        if skill in modelo.wv and vecs_cub:
            cobertura = max(cos(modelo.wv[skill], v) for v in vecs_cub.values())
        else:
            cobertura = 0.0
        filas.append({
            "habilidad": skill, "demanda": freq,
            "cobertura": round(cobertura, 3),
            "es_gap":    cobertura < umbral,
            "peso_gap":  round(freq * (1 - cobertura), 3),
        })
    return pd.DataFrame(filas).sort_values("peso_gap", ascending=False).reset_index(drop=True)

def recomendar_cursos(online, gap_df, tfidf, X, excluir_titulos=None,
                      n_cursos=5, top_k_gap=40, n_candidatos=150):
    excluir_titulos = excluir_titulos or set()
    gap = gap_df[gap_df["es_gap"]].sort_values("peso_gap", ascending=False)
    if gap.empty:
        gap = gap_df.copy()
    gap = gap.head(top_k_gap)

    consulta   = " ".join(gap["habilidad"])
    q          = tfidf.transform([consulta])
    relevancia = cosine_similarity(q, X).ravel()

    M = min(n_candidatos, int((relevancia > 0).sum()))
    if M == 0:
        return pd.DataFrame()

    idx  = np.argsort(relevancia)[::-1][:M]
    cand = online.iloc[idx].copy().reset_index(drop=True)
    cand["relevancia"] = relevancia[idx]
    cand = cand[~cand["titulo"].isin(excluir_titulos)].reset_index(drop=True)
    if cand.empty:
        return pd.DataFrame()

    def norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0 + 1.0

    cand["score"] = norm(cand["relevancia"])
    cand = cand.sort_values("score", ascending=False).reset_index(drop=True)

    peso_skill = dict(zip(gap["habilidad"], gap["peso_gap"]))
    total      = sum(peso_skill.values()) or 1.0
    cand_sets  = [set(t) & set(peso_skill) for t in cand["tokens"]]
    Xc2        = tfidf.transform(cand["doc"])

    elegidos, cubiertas = [], set()
    while len(elegidos) < min(n_cursos, len(cand)):
        mejor_i, mejor_val = None, -1e9
        for i in range(len(cand)):
            if i in elegidos:
                continue
            nuevas    = cand_sets[i] - cubiertas
            cobertura = sum(peso_skill[s] for s in nuevas) / total
            redund    = cosine_similarity(Xc2[i], Xc2[elegidos]).max() if elegidos else 0.0
            val = 0.35 * cand.loc[i, "score"] + 0.45 * cobertura - 0.40 * redund
            if val > mejor_val:
                mejor_val, mejor_i = val, i
        elegidos.append(mejor_i)
        cubiertas |= cand_sets[mejor_i]

    cand = cand.iloc[elegidos].reset_index(drop=True)
    cand["gap_cubierto"] = cand["tokens"].apply(
        lambda t: ", ".join(sorted(set(t) & set(peso_skill))[:8]) or "—"
    )
    return cand[["titulo","descripcion","portal","precio","url","score","gap_cubierto"]]

def skills_de_cursos_completados(titulos, online):
    """Extrae los tokens de los cursos online completados como skills extra."""
    skills = set()
    for titulo in titulos:
        fila = online[online["titulo"] == titulo]
        if not fila.empty:
            skills.update(fila.iloc[0]["tokens"])
    return skills

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Recomendador de Cursos", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    background: #f8fafc;
}
.card h3 { margin: 0 0 4px; font-size: 1.05rem; }
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
}
.pill-c  { background:#dbeafe; color:#1d4ed8; }
.pill-u  { background:#ffedd5; color:#c2410c; }
.pill-g  { background:#f3e8ff; color:#7c3aed; }
.pill-p  { background:#dcfce7; color:#166534; }
.bar-bg  { height:5px; background:#e2e8f0; border-radius:3px; margin:6px 0 2px; }
.bar-fg  { height:5px; background:#6366f1; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Recomendador de Cursos por Brecha de Habilidades")
st.caption("Selecciona tu universidad y carrera, y te recomendamos cursos online para cerrar tu GAP con el mercado laboral.")

# ── Cargar recursos ────────────────────────────────────────────────────────
mallas, ofertas, online, modelo, habilidades_validas, tfidf, X = cargar_todo()

combos        = mallas.groupby(["universidad","carrera"]).size().reset_index(name="n")
universidades = sorted(combos["universidad"].unique())

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Tu perfil")
    univ    = st.selectbox("Universidad", universidades)
    carreras_disp = sorted(combos[combos["universidad"] == univ]["carrera"].tolist())
    carrera = st.selectbox("Carrera", carreras_disp)

    st.divider()
    niveles_opc = sorted(ofertas["Nivel"].dropna().unique().tolist())
    niveles_sel = st.multiselect("Nivel de oferta laboral", niveles_opc, default=niveles_opc)
    n_cursos    = st.slider("Cursos a recomendar", 3, 10, 5)

    st.divider()
    if st.button("🔄 Reiniciar", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Resetear estado si cambia el perfil ───────────────────────────────────
perfil_key = f"{univ}||{carrera}"
if st.session_state.get("_perfil") != perfil_key:
    completados_backup = st.session_state.get("completados", set())
    idx_backup = st.session_state.get("idx_actual", 0)
    
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    
    st.session_state["_perfil"]    = perfil_key
    st.session_state["completados"] = completados_backup  # ← preserva
    st.session_state["idx_actual"]  = idx_backup          # ← preserva

# Inicializar estado
if "completados"    not in st.session_state:
    st.session_state.completados    = set()   # títulos de cursos online completados
if "skills_malla"   not in st.session_state:
    st.session_state.skills_malla   = set()   # skills de malla (no usadas en este flujo)
if "recs"           not in st.session_state:
    st.session_state.recs           = None
if "gap_df"         not in st.session_state:
    st.session_state.gap_df         = None
if "idx_actual"     not in st.session_state:
    st.session_state.idx_actual     = 0       # índice del curso que se está mostrando

# ─────────────────────────────────────────────────────────────────────────────
# CALCULAR / RECALCULAR RECOMENDACIONES
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_pipeline():
    ofr = ofertas.copy()
    if niveles_sel:
        ofr = ofr[ofr["Nivel"].isin(niveles_sel)].reset_index(drop=True)
    ofr_rel = filtrar_ofertas_por_carrera(ofr, carrera, modelo)
    demanda = construir_demanda(ofr_rel)

    # Skills del alumno = tokens de todos los cursos online ya completados
    skills_online = skills_de_cursos_completados(st.session_state.completados, online)

    gap_df = calcular_gap(demanda, sorted(skills_online), modelo)
    recs   = recomendar_cursos(
        online, gap_df, tfidf, X,
        excluir_titulos=st.session_state.completados,
        n_cursos=n_cursos,
    )
    st.session_state.gap_df    = gap_df
    st.session_state.recs      = recs
    st.session_state.idx_actual = 0

# Calcular al arrancar (si no hay resultados aún)
if st.session_state.recs is None:
    with st.spinner("Calculando recomendaciones…"):
        ejecutar_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# MOSTRAR MÉTRICAS DEL GAP
# ─────────────────────────────────────────────────────────────────────────────
gap_df = st.session_state.gap_df
recs   = st.session_state.recs

if gap_df is not None:
    n_gap  = int(gap_df["es_gap"].sum())
    n_comp = len(st.session_state.completados)

    m1, m2, m3 = st.columns(3)
    m1.metric("Skills en el GAP",        n_gap)
    m2.metric("Cursos online completados", n_comp)
    m3.metric("Cursos recomendados",      len(recs) if recs is not None and not recs.empty else 0)

    with st.expander("🔍 Ver top habilidades del GAP"):
        top = gap_df[gap_df["es_gap"]].head(12)[["habilidad","demanda","cobertura","peso_gap"]]
        top.columns = ["Habilidad","Demanda","Cobertura","Peso GAP"]
        st.dataframe(top, hide_index=True, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TARJETAS DE CURSOS — UNA A LA VEZ
# ─────────────────────────────────────────────────────────────────────────────
if recs is None or recs.empty:
    st.success("🎉 ¡No hay más cursos que recomendar para este GAP! Prueba cambiando los filtros.")
    st.stop()

idx     = st.session_state.idx_actual
total   = len(recs)

if idx >= total:
    st.success("🎉 Revisaste todos los cursos recomendados.")
    if st.button("🔁 Calcular nuevas recomendaciones"):
        with st.spinner("Recalculando…"):
            ejecutar_pipeline()
        st.rerun()
    st.stop()

row = recs.iloc[idx]

titulo  = row["titulo"]
portal  = str(row.get("portal",""))
precio  = row.get("precio","No especificado")
url     = str(row.get("url",""))
score   = float(row.get("score", 0))
gap_cub = row.get("gap_cubierto","—")
desc    = str(row.get("descripcion",""))

# Badge de portal
if "coursera" in portal.lower():
    badge_portal = '<span class="pill pill-c">🟦 Coursera</span>'
else:
    badge_portal = '<span class="pill pill-u">🟧 Udemy</span>'

# Badge de precio
if precio.lower() == "gratis":
    badge_precio = '<span class="pill pill-p">🆓 Gratis</span>'
else:
    badge_precio = f'<span class="pill" style="background:#f1f5f9;color:#475569;">💰 {precio}</span>'

link_html = f'<a href="{url}" target="_blank">🔗 Ver curso</a>' if url else ""

pct = int(score * 100)
skills_html = "".join(
    f'<span class="pill pill-g">{s}</span>'
    for s in gap_cub.split(", ") if s and s != "—"
)

# Progreso
st.markdown(f"**Curso {idx + 1} de {total}**")
st.progress((idx + 1) / total)

# Tarjeta
st.markdown(f"""
<div class="card">
  <h3>{titulo}</h3>
  <div style="margin:6px 0 10px;">
    {badge_portal} {badge_precio} {link_html}
  </div>
  <div class="bar-bg"><div class="bar-fg" style="width:{pct}%"></div></div>
  <span style="font-size:0.75rem;color:#64748b;">Relevancia para tu GAP: {score:.2f}</span>
  <p style="margin:10px 0 6px;font-size:0.85rem;color:#334155;">{desc[:350]}…</p>
  <div style="margin-top:8px;"><strong style="font-size:0.8rem;">Habilidades que cubre:</strong><br/>{skills_html if skills_html else '<span style="color:#94a3b8;font-size:0.8rem;">—</span>'}</div>
</div>
""", unsafe_allow_html=True)

# ── Botones de acción ──────────────────────────────────────────────────────
st.markdown("#### ¿Ya conoces este curso o lo completaste?")
col_si, col_no = st.columns(2)

with col_si:
    if st.button("✅ Ya lo sé / lo completé → siguiente", use_container_width=True, type="primary"):
        st.session_state.completados.add(titulo)
        with st.spinner("Actualizando recomendaciones…"):
            ejecutar_pipeline()
        # Restaurar índice después del recálculo, excluyendo el curso recién completado
        nuevo_total = len(st.session_state.recs) if st.session_state.recs is not None else 0
        st.session_state.idx_actual = min(st.session_state.idx_actual, nuevo_total)
        st.rerun()

with col_no:
    if st.button("➡️ No lo conozco → guardar y ver siguiente", use_container_width=True):
        st.session_state.idx_actual += 1
        st.rerun()

# ── Lista de completados ───────────────────────────────────────────────────
if st.session_state.completados:
    st.divider()
    with st.expander(f"📋 Cursos completados en esta sesión ({len(st.session_state.completados)})"):
        for t in sorted(st.session_state.completados):
            st.markdown(f"- ✅ {t}")
