import os, re, ast, pickle, unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse
import streamlit as st
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
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

PORTAL_EMOJI = {"Coursera": "🟦", "Udemy": "🟧"}

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE TEXTO  (idénticas al notebook)
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
SEMILLAS_TECH  = [
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
# VECTORES Y SIMILITUD
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
# CARGA Y ENTRENAMIENTO  (cacheado: solo corre una vez por sesión)
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

@st.cache_resource(show_spinner="Cargando datos y modelos…")
def cargar_todo():
    # Mallas
    mallas = pd.read_csv(RUTA_MALLAS).rename(columns={"Unnamed: 0": "id"})
    mallas["tokens_nombre"] = mallas["tokens"].apply(
        lambda v: limpiar_tokens(parse_lista(v), stop=STOP_CURSO)
    )
    mallas["es_no_tech"] = mallas["tokens_nombre"].apply(
        lambda t: bool(set(t) & TOKENS_NO_TECH)
    )

    # Ofertas
    ofertas = pd.read_csv(RUTA_OFERTAS)
    ofertas["hab_tokens"] = ofertas["habilidades"].apply(parse_lista)
    ofertas["hab_tokens"] = ofertas["hab_tokens"].apply(
        lambda l: limpiar_tokens(l, stop=STOP_OFERTAS)
    )
    ofertas = ofertas[ofertas["hab_tokens"].map(len) > 0].reset_index(drop=True)

    # Cursos online
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
    def entrenar_w2v():
        corpus  = list(online["tokens"]) + list(ofertas["hab_tokens"]) + list(mallas["tokens_nombre"])
        corpus  = [d for d in corpus if d]
        return Word2Vec(sentences=corpus, vector_size=100, window=5,
                        min_count=2, sg=1, workers=4, epochs=20, seed=42)

    if os.path.exists(RUTA_W2V):
        modelo = Word2Vec.load(RUTA_W2V)
    else:
        modelo = entrenar_w2v()
        modelo.save(RUTA_W2V)

    # Filtro 2: similitud coseno con perfil tech
    sem_en_vocab = [s for s in SEMILLAS_TECH if s in modelo.wv]
    v_tech = np.mean([modelo.wv[s] for s in sem_en_vocab], axis=0)
    cand_f1 = mallas[~mallas["es_no_tech"] & (mallas["tokens_nombre"].map(len) > 0)].copy()
    cand_f1["sim_tech"] = cand_f1["tokens_nombre"].apply(
        lambda t: cos(vec(t, modelo), v_tech)
    )
    cursos_tech = set(cand_f1[cand_f1["sim_tech"] >= UMBRAL_SIM_TECH]["curso"])
    mallas["es_tech"] = mallas["curso"].isin(cursos_tech)

    # Habilidades válidas
    freq_hab = Counter()
    for toks in ofertas["hab_tokens"]:
        freq_hab.update(set(toks))
    habilidades_validas = [h for h, n in freq_hab.items() if n >= 5 and h in modelo.wv]

    # Skills predichos
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
# PIPELINE (idéntico al notebook)
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
    """
    TF-IDF + coseno + MMR.
    excluir_titulos: set de títulos que el usuario ya marcó como completados.
    """
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

    # Excluir cursos ya completados
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

# ─────────────────────────────────────────────────────────────────────────────
# APP  STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recomendador de Cursos",
    page_icon="🎓",
    layout="wide",
)

# CSS mínimo
st.markdown("""
<style>
.card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.card-done {
    background: #f0fdf4;
    border-color: #86efac;
    opacity: 0.75;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-coursera { background:#dbeafe; color:#1d4ed8; }
.badge-udemy    { background:#ffedd5; color:#c2410c; }
.badge-gap      { background:#f3e8ff; color:#7c3aed; }
.score-bar-wrap { height:6px; background:#e2e8f0; border-radius:3px; margin:6px 0 4px; }
.score-bar      { height:6px; background:#6366f1; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# Título
st.title("🎓 Recomendador de Cursos por Brecha de Habilidades")
st.caption("Encuentra los cursos online que cierran el GAP entre lo que sabes y lo que pide el mercado.")

# ── Carga de recursos ──────────────────────────────────────────────────────
mallas, ofertas, online, modelo, habilidades_validas, tfidf, X = cargar_todo()

combos = (mallas.groupby(["universidad","carrera"])
          .size().reset_index(name="n_cursos"))
universidades = sorted(combos["universidad"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: Selección de perfil
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Tu perfil")

    univ = st.selectbox("Universidad", universidades)
    carreras_disp = sorted(combos[combos["universidad"] == univ]["carrera"].tolist())
    carrera = st.selectbox("Carrera", carreras_disp)

    st.divider()
    niveles_opciones = sorted(ofertas["Nivel"].dropna().unique().tolist())
    niveles_sel = st.multiselect(
        "Nivel de oferta laboral",
        options=niveles_opciones,
        default=niveles_opciones,
        help="Filtra las ofertas que definen la demanda del mercado.",
    )
    n_cursos = st.slider("Cursos a recomendar", 3, 10, 5)

    st.divider()
    if st.button("🔄 Reiniciar todo", use_container_width=True):
        for k in ["skills_marcadas","cursos_completados","recs","gap_df","info"]:
            st.session_state.pop(k, None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────
perfil_key = f"{univ}||{carrera}"
if st.session_state.get("_perfil_key") != perfil_key:
    for k in ["skills_marcadas","cursos_completados","recs","gap_df","info"]:
        st.session_state.pop(k, None)
    st.session_state["_perfil_key"] = perfil_key

if "skills_marcadas"    not in st.session_state:
    st.session_state.skills_marcadas    = set()
if "cursos_completados" not in st.session_state:
    st.session_state.cursos_completados = set()

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: Malla universitaria — checkboxes de skills
# ─────────────────────────────────────────────────────────────────────────────
sub_malla = (mallas[(mallas["universidad"] == univ) & (mallas["carrera"] == carrera)]
             .copy())
sub_tech  = sub_malla[sub_malla["skills_predichos"].map(len) > 0].sort_values(
    ["ciclo","curso"] if "ciclo" in sub_malla.columns else ["curso"]
)

st.subheader("📚 Paso 1 · Marca los cursos que ya dominas")
st.caption("Elige los cursos cuyas habilidades ya conoces. Puedes seleccionar por ciclo o individualmente.")

if sub_tech.empty:
    st.warning("No hay cursos técnicos con skills predichos para esta carrera.")
    st.stop()

# Agrupar por ciclo si existe
tiene_ciclo = "ciclo" in sub_tech.columns
grupos = sub_tech.groupby("ciclo") if tiene_ciclo else [("", sub_tech)]

nuevas_skills: set = set()

with st.expander("Ver / ocultar malla completa", expanded=True):
    for ciclo, grupo in grupos:
        if tiene_ciclo:
            col_header, col_todo = st.columns([5, 1])
            with col_header:
                st.markdown(f"**Ciclo {ciclo}**")
            with col_todo:
                todos_key = f"ciclo_todo_{ciclo}"
                if st.checkbox("todos", key=todos_key, label_visibility="collapsed"):
                    for _, r in grupo.iterrows():
                        nuevas_skills.update(r["skills_predichos"])

        for _, row in grupo.iterrows():
            label = f"{row['curso']}  —  `{'`, `'.join(row['skills_predichos'])}`"
            checked = st.checkbox(
                label,
                key=f"curso_{row['curso']}",
                value=(row["curso"] in {
                    c for c in st.session_state.get("_cursos_marcados", set())
                }),
            )
            if checked:
                nuevas_skills.update(row["skills_predichos"])
                st.session_state.setdefault("_cursos_marcados", set()).add(row["curso"])
            else:
                st.session_state.setdefault("_cursos_marcados", set()).discard(row["curso"])

st.session_state.skills_marcadas = nuevas_skills

# Resumen de skills dominadas
n_dom = len(st.session_state.skills_marcadas)
if n_dom:
    st.success(f"✅ {n_dom} skill{'s' if n_dom>1 else ''} marcada{'s' if n_dom>1 else ''} como dominada{'s' if n_dom>1 else ''}.")
else:
    st.info("Aún no marcaste ningún curso. Si no marcas nada, el GAP incluirá todas las habilidades demandadas.")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: Recomendar
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
col_btn, col_info = st.columns([2, 5])
with col_btn:
    recomendar_btn = st.button("🚀 Recomendar cursos", type="primary", use_container_width=True)

if recomendar_btn:
    with st.spinner("Calculando GAP y buscando cursos…"):
        ofr = ofertas.copy()
        if niveles_sel:
            ofr = ofr[ofr["Nivel"].isin(niveles_sel)].reset_index(drop=True)
        ofr_rel  = filtrar_ofertas_por_carrera(ofr, carrera, modelo)
        demanda  = construir_demanda(ofr_rel)
        gap_df   = calcular_gap(demanda, sorted(st.session_state.skills_marcadas), modelo)
        recs     = recomendar_cursos(
            online, gap_df, tfidf, X,
            excluir_titulos=st.session_state.cursos_completados,
            n_cursos=n_cursos,
        )
        st.session_state.recs    = recs
        st.session_state.gap_df  = gap_df
        st.session_state.info    = {
            "skills_dominadas":       n_dom,
            "ofertas_relevantes":     len(ofr_rel),
            "habilidades_demandadas": len(demanda),
            "gap":                    int(gap_df["es_gap"].sum()),
        }

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: Mostrar resultados
# ─────────────────────────────────────────────────────────────────────────────
if "recs" in st.session_state and st.session_state.recs is not None:
    recs   = st.session_state.recs
    gap_df = st.session_state.gap_df
    info   = st.session_state.info

    st.subheader("📊 Resumen del GAP")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skills dominadas",        info["skills_dominadas"])
    m2.metric("Ofertas analizadas",       info["ofertas_relevantes"])
    m3.metric("Habilidades demandadas",   info["habilidades_demandadas"])
    m4.metric("Skills en el GAP",         info["gap"])

    with st.expander("🔍 Ver detalle del GAP (top 15 habilidades)"):
        top_gap = gap_df[gap_df["es_gap"]].head(15)[
            ["habilidad","demanda","cobertura","peso_gap"]
        ].rename(columns={
            "habilidad": "Habilidad", "demanda": "Demanda",
            "cobertura": "Cobertura", "peso_gap": "Peso GAP",
        })
        st.dataframe(top_gap, use_container_width=True, hide_index=True)

    st.divider()

    if recs.empty:
        st.warning("No se encontraron cursos para este GAP (prueba con menos cursos completados o cambia los filtros).")
    else:
        completados_actualizados = set(st.session_state.cursos_completados)
        recalcular = False

        st.subheader(f"🎯 Paso 2 · Cursos recomendados")
        st.caption("Marca un curso como **completado** para actualizar las recomendaciones automáticamente.")

        for _, row in recs.iterrows():
            titulo   = row["titulo"]
            portal   = str(row.get("portal", ""))
            precio   = row.get("precio", "No especificado")
            url      = row.get("url", "")
            score    = float(row.get("score", 0))
            gap_cub  = row.get("gap_cubierto", "—")
            desc     = str(row.get("descripcion", ""))[:220] + "…"

            ya_completado = titulo in st.session_state.cursos_completados
            card_class    = "card card-done" if ya_completado else "card"

            badge_portal = (
                f'<span class="badge badge-coursera">🟦 Coursera</span>' if "coursera" in portal.lower()
                else f'<span class="badge badge-udemy">🟧 Udemy</span>'
            )
            pct = int(score * 100)
            link_html = (
                f'<a href="{url}" target="_blank" style="font-size:0.8rem;">🔗 Ver curso</a>'
                if url else ""
            )

            st.markdown(f"""
<div class="{card_class}">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <div style="flex:1">
      <strong style="font-size:1rem;">{titulo}</strong><br/>
      {badge_portal}
      <span style="font-size:0.8rem; color:#64748b;">💰 {precio}</span>
      &nbsp;{link_html}
      <div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%"></div></div>
      <span style="font-size:0.75rem; color:#64748b;">Relevancia: {score:.2f}</span>
      &nbsp;&nbsp;
      <span class="badge badge-gap">🎯 {gap_cub}</span>
    </div>
  </div>
  <p style="font-size:0.82rem; color:#475569; margin: 0.5rem 0 0;">{desc}</p>
</div>
""", unsafe_allow_html=True)

            completado_check = st.checkbox(
                f"✅ Ya completé este curso",
                key=f"completado_{titulo}",
                value=ya_completado,
            )
            if completado_check and titulo not in st.session_state.cursos_completados:
                st.session_state.cursos_completados.add(titulo)
                recalcular = True
            elif not completado_check and titulo in st.session_state.cursos_completados:
                st.session_state.cursos_completados.discard(titulo)
                recalcular = True

        # Si el usuario marcó/desmarcó algo → recalcular automáticamente
        if recalcular:
            with st.spinner("Actualizando recomendaciones…"):
                ofr = ofertas.copy()
                if niveles_sel:
                    ofr = ofr[ofr["Nivel"].isin(niveles_sel)].reset_index(drop=True)
                ofr_rel  = filtrar_ofertas_por_carrera(ofr, carrera, modelo)
                demanda  = construir_demanda(ofr_rel)

                # Skills que el alumno ya domina = malla + skills de cursos online completados
                skills_extra = set()
                for titulo_comp in st.session_state.cursos_completados:
                    fila = online[online["titulo"] == titulo_comp]
                    if not fila.empty:
                        skills_extra.update(fila.iloc[0]["tokens"])

                skills_totales = st.session_state.skills_marcadas | skills_extra

                gap_df_new = calcular_gap(demanda, sorted(skills_totales), modelo)
                recs_new   = recomendar_cursos(
                    online, gap_df_new, tfidf, X,
                    excluir_titulos=st.session_state.cursos_completados,
                    n_cursos=n_cursos,
                )
                st.session_state.recs   = recs_new
                st.session_state.gap_df = gap_df_new
                st.session_state.info["skills_dominadas"] = len(skills_totales)
                st.session_state.info["gap"] = int(gap_df_new["es_gap"].sum())
            st.rerun()

        if st.session_state.cursos_completados:
            st.divider()
            st.markdown(f"**Cursos completados en esta sesión ({len(st.session_state.cursos_completados)}):**")
            for t in sorted(st.session_state.cursos_completados):
                st.markdown(f"- ~~{t}~~")
