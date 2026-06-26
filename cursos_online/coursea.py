"""
====================================================================
COURSERA SCRAPER v3.2 - MÁXIMA COBERTURA (API + Playwright)
CON CHECKPOINT/RESUME + TOLERANCIA A CRASHES + FILTRO IDIOMA/TIPO
====================================================================
CAMBIOS RESPECTO A v3.1:

  1) RESUME REAL EN ETAPA 2
     Si el CSV ya existe, se carga al arrancar. Etapa 1 (API) se
     vuelve a correr siempre (es rápida e idempotente) pero MERGEA
     con lo que ya tenías en vez de pisarlo: si un curso ya tiene
     "headline" relleno (o sea, Etapa 2 ya lo procesó), no se vuelve
     a visitar.

  2) TOLERANCIA A CRASHES DE PLAYWRIGHT
     Si tu PC se suspende, se corta la red, o el navegador se cierra
     solo a mitad de Etapa 2 (error típico: "Target page, context or
     browser has been closed"), antes esto mataba TODO el script.
     Ahora se detecta, se reinicia el navegador, y se sigue desde el
     curso siguiente sin perder lo ya guardado.

  3) CTRL+C CONTROLADO
     Podés pausar en cualquier momento; guarda antes de salir.

  4) FILTRO DE IDIOMA Y TIPO
     Como pediste, el dataset final solo incluye:
       - tipo_url == "curso" o "especializacion" (se excluyen
         "proyecto" / Guided Projects, que es lo más distinto a
         Udemy y lo que menos info comparable trae).
       - idioma que contenga "en" o "es" (inglés o español).
     OJO: el tipo real solo se sabe después de visitar la página en
     Etapa 2 (la API no lo distingue), así que el filtro de idioma
     se aplica ANTES de Etapa 2 (ahorra visitas), y el filtro de tipo
     se aplica DESPUÉS (al armar el CSV final), porque no hay forma
     de saberlo sin visitar.

INSTALACIÓN:
  pip install playwright pandas requests
  playwright install chromium

PARA PROBAR PRIMERO:
  LIMITE_CURSOS   = 100
  LIMITE_DETALLES = 15

PARA PAUSAR: Ctrl+C en cualquier momento.
PARA REANUDAR: volver a ejecutar el mismo script.
====================================================================
"""

import time
import random
import re
import sys
import requests
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
OUTPUT_CSV      = "coursera_cursos_v5.csv"
PAGE_SIZE       = 100
LIMITE_CURSOS   = None
SCRAPE_DETALLES = True
LIMITE_DETALLES = None
HEADLESS        = True

# Idiomas que queremos conservar (se busca como substring, insensible a mayúsc.)
IDIOMAS_PERMITIDOS = ["en", "es"]

# Si True, al arrancar imprime TODAS las claves crudas que trae el primer
# elemento de la API (y guarda el JSON crudo en debug_api_raw.json). Sirve
# para diagnosticar campos que vienen vacíos (rating, nivel, precio, etc.)
# sin tener que adivinar nombres de campo.
DEBUG_API_FIELDS = True

# Si N páginas seguidas fallan por crash/cierre de browser, reiniciamos sesión
MAX_FALLOS_SEGUIDOS   = 5
ESPERA_TRAS_FALLO_S    = 60

URL_CATALOG = "https://api.coursera.org/api/courses.v1"
# Pedimos un set más amplio de campos: algunos son los que ya usábamos,
# otros son nombres alternativos que la API de Coursera ha usado en
# distintas versiones/respuestas (rating, nivel y precio no siempre
# están bajo el mismo nombre). Pedir de más no rompe nada: si un campo
# no existe para ese curso, la API simplemente no lo incluye.
FIELDS_V1 = (
    "description,instructorIds,partnerIds,workload,"
    "primaryLanguages,courseType,domainTypes,"
    "avgProductRating,numProductRatings,difficultyLevel,"
    "isCourseFree,skills,"
    # candidatos alternativos:
    "rating,numRatings,averageRating,avgRating,"
    "courseLevel,level,"
    "isFree,free,pricing,"
    "primarySkills,relatedSkills,skillTags"
)
INCLUDES_V1 = "instructorIds,partnerIds"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ══════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════

def limpiar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)
    return " ".join(texto.split())

def limpiar_lista(valores) -> str:
    if not valores:
        return ""
    if isinstance(valores, list):
        return " | ".join(str(v).strip() for v in valores if v)
    return str(valores).strip()

def normalizar_nivel(nivel_raw: str) -> str:
    mapa = {
        "BEGINNER":     "Principiante",
        "INTERMEDIATE": "Intermedio",
        "ADVANCED":     "Avanzado",
        "MIXED":        "Mixto / Todos los niveles",
    }
    return mapa.get(str(nivel_raw).upper(), nivel_raw or "")

def normalizar_precio(is_free) -> str:
    if is_free is True:  return "Gratis"
    if is_free is False: return "De pago (Coursera Plus)"
    return ""

def detectar_tipo_url(url: str) -> str:
    if not url:
        return "curso"
    if "/projects/" in url:
        return "proyecto"
    if "/specializations/" in url:
        return "especializacion"
    return "curso"

def idioma_permitido(idioma_texto: str) -> bool:
    """True si el idioma contiene 'en' o 'es' (inglés/español)."""
    if not idioma_texto:
        # sin dato de idioma: lo dejamos pasar, se filtra mejor después
        return True
    t = idioma_texto.lower()
    return any(re.search(rf"\b{idi}\b", t) or idi in t for idi in IDIOMAS_PERMITIDOS)


# ══════════════════════════════════════════════════════════════════
# PERSISTENCIA / RESUME
# ══════════════════════════════════════════════════════════════════

def cargar_csv_existente() -> pd.DataFrame:
    path = Path(OUTPUT_CSV)
    if path.exists():
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
            print(f"📂 CSV existente encontrado: {len(df):,} filas cargadas (retomando)")
            return df
        except Exception as e:
            print(f"⚠ No se pudo leer el CSV existente ({e}), arranco de cero")
    return pd.DataFrame()


def guardar_csv(registros: list[dict]):
    pd.DataFrame(registros).to_csv(OUTPUT_CSV, sep=";", index=False, encoding="utf-8-sig")


def merge_con_existente(nuevos: list[dict], existente: pd.DataFrame) -> list[dict]:
    """
    Combina lo recién bajado de la API con lo que ya había en el CSV.
    Si un curso ya tenía 'tipo_url' relleno (Etapa 2 ya lo procesó, lo
    haya marcado como curso, especializacion o proyecto), conservamos
    esos datos de detalle en vez de pisarlos con los vacíos que trae
    la API en una nueva corrida.
    """
    if existente.empty:
        return nuevos

    existente_por_id = {
        row["id_coursera"]: row.to_dict()
        for _, row in existente.iterrows()
        if row.get("id_coursera")
    }

    fusionados = []
    for curso in nuevos:
        prev = existente_por_id.get(curso.get("id_coursera"))
        if prev and (prev.get("tipo_url") or "").strip():
            # ya tenía detalle scrapeado: conservamos esos campos de Etapa 2
            campos_detalle = [
                "tipo_url", "headline", "num_inscritos", "horas_semana",
                "duracion_semanas", "duracion_total", "cronograma",
                "tipo_programa", "que_aprenderas", "habilidades",
                "herramientas", "requisitos", "temario", "temario_stats", "url",
                # estos 3 pueden venir de la página (Playwright) cuando la
                # API no los trae; si ya los teníamos, no los pisamos.
                "rating", "num_reseñas", "precio",
            ]
            for c in campos_detalle:
                if c in prev:
                    curso[c] = prev[c]
        fusionados.append(curso)
    return fusionados


# ══════════════════════════════════════════════════════════════════
# ETAPA 1 — API
# ══════════════════════════════════════════════════════════════════

def fetch_pagina_catalogo(start: int) -> dict:
    params = {
        "start": start, "limit": PAGE_SIZE,
        "fields": FIELDS_V1, "includes": INCLUDES_V1,
    }
    try:
        r = requests.get(URL_CATALOG, params=params, headers=HEADERS, timeout=20)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        print(f"  ❌ Error de conexión (start={start}): {e}")
        return {}

def primer_valor(item: dict, *claves):
    """Devuelve el primer valor no-vacío entre varios nombres de campo posibles."""
    for clave in claves:
        v = item.get(clave)
        if v is not None and v != "" and v != []:
            return v
    return None


def procesar_elemento(item: dict, instructores: dict, partners: dict) -> dict:
    slug = item.get("slug", "")
    url  = f"https://www.coursera.org/learn/{slug}" if slug else ""

    instructor_nombres = [instructores.get(i, "") for i in (item.get("instructorIds") or [])]
    partner_nombres    = [partners.get(p, "")     for p in (item.get("partnerIds")    or [])]

    rating_raw  = primer_valor(item, "avgProductRating", "rating", "averageRating", "avgRating")
    rating      = f"{rating_raw:.1f}" if isinstance(rating_raw, (int, float)) else (str(rating_raw) if rating_raw else "")

    reseñas_raw = primer_valor(item, "numProductRatings", "numRatings")
    num_reseñas = f"{reseñas_raw:,}" if isinstance(reseñas_raw, int) else (str(reseñas_raw) if reseñas_raw else "")

    nivel_raw = primer_valor(item, "difficultyLevel", "courseLevel", "level")
    is_free   = primer_valor(item, "isCourseFree", "isFree", "free")

    skills_raw = primer_valor(item, "skills", "primarySkills", "relatedSkills", "skillTags")

    domain_types = item.get("domainTypes") or []
    categorias   = limpiar_lista([
        dt.get("domainId", "") + ("/" + dt.get("subdomainId", "") if dt.get("subdomainId") else "")
        for dt in domain_types if isinstance(dt, dict)
    ])

    return {
        "titulo":           item.get("name", ""),
        "tipo_url":         "",
        "headline":         "",
        "instructor":       " | ".join(filter(None, instructor_nombres)),
        "universidad":      " | ".join(filter(None, partner_nombres)),
        "rating":           rating,
        "num_reseñas":      num_reseñas,
        "num_inscritos":    "",
        "horas_semana":     "",
        "duracion_semanas": "",
        "duracion_total":   "",
        "duracion_api":     item.get("workload", ""),
        "cronograma":       "",
        "nivel":            normalizar_nivel(nivel_raw) if nivel_raw is not None else "",
        "precio":           normalizar_precio(is_free) if is_free is not None else "",
        "url":              url,
        "idioma":           limpiar_lista(item.get("primaryLanguages") or []),
        "descripcion":      item.get("description", ""),
        "objetivos_api":    limpiar_lista(skills_raw) if skills_raw else "",
        "que_aprenderas":   "",
        "habilidades":      "",
        "herramientas":     "",
        "requisitos":       "",
        "temario":          "",
        "temario_stats":    "",
        "tipo_curso_api":   item.get("courseType", ""),
        "categorias":       categorias,
        "slug":             slug,
        "id_coursera":      item.get("id", ""),
    }

def etapa1_api() -> list[dict]:
    todos = []
    start = 0
    pagina = 0
    total_api = None

    print("=" * 65)
    print("ETAPA 1 — API pública courses.v1")
    print("=" * 65)

    while True:
        pagina += 1
        data = fetch_pagina_catalogo(start)
        if not data:
            break

        elementos = data.get("elements", [])
        paging    = data.get("paging", {})
        if total_api is None:
            total_api = paging.get("total")

        if not elementos:
            print(f"  🏁 Sin más elementos en página {pagina}.")
            break

        if DEBUG_API_FIELDS and pagina == 1:
            import json as _json
            primero = elementos[0]
            print(f"\n  🔬 DEBUG — claves crudas del primer curso ('{primero.get('name','')[:40]}'):")
            for k in sorted(primero.keys()):
                v = primero[k]
                v_str = str(v)
                if len(v_str) > 80:
                    v_str = v_str[:80] + "..."
                print(f"      {k:22s} = {v_str}")
            try:
                Path("debug_api_raw.json").write_text(
                    _json.dumps(elementos[0], ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"  💾 JSON crudo del primer curso guardado en debug_api_raw.json\n")
            except Exception:
                pass

        linked       = data.get("linked", {})
        instructores = {i["id"]: i.get("fullName", "") for i in linked.get("instructors.v1", [])}
        partners     = {p["id"]: p.get("name", "")    for p in linked.get("partners.v1",    [])}

        for item in elementos:
            todos.append(procesar_elemento(item, instructores, partners))

        print(
            f"  📄 Pág {pagina}: {len(elementos)} cursos "
            f"(acumulado: {len(todos):,}"
            + (f" / total: {total_api:,}" if total_api else "") + ")"
        )

        if LIMITE_CURSOS and len(todos) >= LIMITE_CURSOS:
            print(f"  🛑 Límite de {LIMITE_CURSOS:,} alcanzado")
            break

        if total_api and start + PAGE_SIZE >= total_api:
            print(f"  🏁 Catálogo completo ({total_api:,} cursos)")
            break

        start += PAGE_SIZE
        time.sleep(0.3)

    return todos


# ══════════════════════════════════════════════════════════════════
# ETAPA 2 — Playwright
# ══════════════════════════════════════════════════════════════════

JS_CURSO = r"""
() => {
    const txt  = (sel, root) => { const e=(root||document).querySelector(sel); return e?e.textContent.trim():''; };
    const txts = (sel, root) => Array.from((root||document).querySelectorAll(sel)).map(e=>e.textContent.trim()).filter(Boolean);

    const url_real = window.location.href;

    // ── Headline ──────────────────────────────────────────────────
    // No hay un selector dedicado para "headline" en el rediseño actual;
    // lo que existía antes era principalmente la frase de "este curso es
    // parte de la especialización X". La buscamos por texto en vez de
    // adivinar una clase.
    let headline = '';
    const todoElTexto = Array.from(document.querySelectorAll('p,span,div'))
        .filter(el => el.children.length === 0)
        .map(el => el.textContent.trim());
    headline = todoElTexto.find(t =>
        t.length > 10 && t.length < 300 &&
        /(este curso es parte de|this course is part of|programa especializado|specialization)/i.test(t)
    ) || '';

    // ── num_inscritos ────────────────────────────────────────────
    // CONFIRMADO (jun 2026): Coursera ya NO muestra el conteo de
    // inscritos en la página de curso rediseñada. No existe en el DOM.
    // Lo dejamos siempre vacío para no perder tiempo buscando algo que
    // no está.
    const num_inscritos = '';

    // ── Bloque "key-information" (rating, reseñas, duración, cronograma) ──
    // Confirmado con HTML real: data-e2e="key-information" agrupa varios
    // bloques .css-dwgey1, cada uno con un texto principal corto.
    const keyInfo = document.querySelector('[data-e2e="key-information"]');
    const statItems = keyInfo
        ? Array.from(keyInfo.querySelectorAll('div'))
              .filter(el => el.children.length === 0)
              .map(el => el.textContent.trim())
              .filter(Boolean)
        : [];

    const buscarStat = (...patrones) =>
        statItems.find(t => patrones.some(p => new RegExp(p, 'i').test(t))) || '';

    let duracion_semanas = buscarStat('\\d+\\s*(semanas?|weeks?)\\b');
    let horas_semana     = buscarStat('\\d+[\\s-]*\\d*\\s*(horas?|hours?)\\s*(a la semana|por semana|/?\\s*semana|a week|/?\\s*week)');
    let cronograma       = buscarStat('cronograma','flexible','self.?paced','fixed deadlines','fechas fijas','propio ritmo');
    let duracion_total   = buscarStat('^\\d+\\s*(horas?|hours?)\\s*(para completar|to complete)?$', '^\\d+\\s*(minutos?|minutes?)');

    // Rating: vive como aria-label="X estrellas"/"X stars" dentro de key-information
    let rating_pagina = '';
    let num_reseñas_pagina = '';
    if (keyInfo) {
        const ratingEl = keyInfo.querySelector('[role="img"][aria-roledescription="rating"]');
        if (ratingEl) {
            const m = (ratingEl.getAttribute('aria-label') || ratingEl.textContent || '').match(/[\d.,]+/);
            if (m) rating_pagina = m[0];
            // la reseña-count suele ser el <p> hermano siguiente con texto tipo "13 reseña(s)"/"reviews"
            let sib = ratingEl.closest('div')?.parentElement?.querySelector('p');
            if (sib && /reseñ|review/i.test(sib.textContent)) {
                const m2 = sib.textContent.match(/[\d.,]+/);
                if (m2) num_reseñas_pagina = m2[0];
            }
        }
    }

    // ── Precio (no viene de la API) ──────────────────────────────
    // Se infiere del texto cerca del botón de inscripción.
    let precio_pagina = '';
    const enrollArea = document.querySelector('[data-e2e="enroll-button"]')?.closest('div')?.parentElement;
    if (enrollArea) {
        const t = enrollArea.textContent || '';
        if (/incluido con|included with|coursera plus/i.test(t)) precio_pagina = 'Incluido con Coursera Plus';
        else if (/gratis|free/i.test(t)) precio_pagina = 'Gratis';
    }

    // ── Tipo de programa / breadcrumb ────────────────────────────
    const tipo_programa =
        txt('[data-testid="product-type-badge"]') ||
        (() => {
            const bc = document.querySelectorAll('[class*="Breadcrumb"] li, [class*="breadcrumb"] li');
            return bc.length ? bc[bc.length-1].textContent.trim() : '';
        })();

    // ── Qué aprenderás ────────────────────────────────────────────
    const queAprenderasSection = (() => {
        const heads = Array.from(document.querySelectorAll('h2,h3,h4'));
        const h = heads.find(h => /qué aprenderás|what you('ll| will) learn|learning objectives/i.test(h.textContent));
        if (!h) return null;
        let parent = h.parentElement;
        for (let i = 0; i < 4; i++) {
            if (parent && parent.querySelectorAll('li').length > 0) return parent;
            parent = parent?.parentElement;
        }
        return h.parentElement;
    })();
    const que_aprenderas = queAprenderasSection
        ? txts('li', queAprenderasSection).join(' | ')
        : '';

    // ── Habilidades / Herramientas ───────────────────────────────
    // CONFIRMADO con HTML real: ul[data-testid="skills-section"] y
    // ul[data-testid="tools-section"]. Esto también es lo que reemplaza
    // a "objetivos_api" de la API (que ya no manda el campo skills).
    const habilidades = txts('ul[data-testid="skills-section"] li, [data-testid^="skill-tag-"]')
        .filter((v, i, a) => a.indexOf(v) === i)
        .join(' | ');

    const herramientas = txts('ul[data-testid="tools-section"] li, [data-testid^="tool-tag-"]')
        .filter((v, i, a) => a.indexOf(v) === i)
        .join(' | ');

    // ── Requisitos ───────────────────────────────────────────────
    // Confirmado: en varios tipos de curso esta sección directamente
    // no existe en la página (no es un selector roto, el dato no está).
    const reqSection = (() => {
        const heads = Array.from(document.querySelectorAll('h2,h3,h4'));
        const h = heads.find(h => /requisito|requirement|conocimientos previos|prerequisites/i.test(h.textContent));
        if (!h) return null;
        let parent = h.parentElement;
        for (let i = 0; i < 4; i++) {
            if (parent && parent.querySelectorAll('li,p').length > 0) return parent;
            parent = parent?.parentElement;
        }
        return h.parentElement;
    })();
    const requisitos = reqSection ? txts('li, p', reqSection).join(' | ') : '';

    // ── Temario (módulos) ─────────────────────────────────────────
    // CONFIRMADO con HTML real: cada módulo vive en
    // [data-testid="accordion-item"], con el título en un <h3> limpio
    // (sin el texto "Qué incluye" pegado, eso es un <h4> aparte).
    const temario = txts('[data-testid="accordion-item"] h3')
        .filter((v, i, a) => a.indexOf(v) === i)
        .join(' | ');

    // temario_stats: el resumen "N módulos" que aparece en key-information,
    // más el detalle por módulo (ej: "Módulo 1 • 3 horas para finalizar")
    let temario_stats = '';
    if (keyInfo) {
        const m = statItems.find(t => /\d+\s*(módulos?|modules?)\b/i.test(t));
        if (m) temario_stats = m;
    }
    const statsPorModulo = txts('[data-testid="accordion-item"] [class*="css-aovwea"], [data-testid="accordion-item"] >div >div >div >div:nth-child(2)')
        .filter((v, i, a) => a.indexOf(v) === i);
    if (statsPorModulo.length) {
        temario_stats = (temario_stats ? temario_stats + ' || ' : '') + statsPorModulo.join(' | ');
    }

    return {
        url_real,
        headline,
        num_inscritos,
        horas_semana,
        duracion_semanas,
        duracion_total,
        cronograma,
        rating_pagina,
        num_reseñas_pagina,
        precio_pagina,
        tipo_programa,
        que_aprenderas,
        habilidades,
        herramientas,
        requisitos,
        temario,
        temario_stats,
    };
}
"""


def extraer_detalle_curso(page, url: str) -> dict:
    # Si algo falla acá (timeout, browser cerrado, etc.), la excepción
    # se propaga y la maneja el llamador (etapa2_playwright), que decide
    # si reintentar, reiniciar sesión, o dejar el curso pendiente.
    page.goto(url, wait_until="domcontentloaded", timeout=25000)
    page.wait_for_timeout(random.uniform(2000, 3200))

    resultado = page.evaluate(JS_CURSO)
    return {k: limpiar_texto(str(v)) if v else "" for k, v in resultado.items()}


def crear_browser(p):
    browser = p.chromium.launch(
        headless=HEADLESS,
        channel="chrome",
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        locale="es-ES",
    )
    page = context.new_page()
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    """)
    return browser, context, page


def etapa2_playwright(todos: list[dict]) -> None:
    # Solo procesamos los que NO tienen tipo_url todavía (resume real).
    # OJO: usamos tipo_url y no headline como marcador de "ya procesado",
    # porque headline ahora suele venir vacío legítimamente (la mayoría
    # de los cursos no son parte de una especialización) y eso rompería
    # el resume si lo usáramos como marcador.
    pendientes = [
        c for c in todos
        if not (c.get("tipo_url") or "").strip()
        and idioma_permitido(c.get("idioma", ""))
    ]
    if LIMITE_DETALLES is not None:
        pendientes = pendientes[:LIMITE_DETALLES]

    print(f"\n{'='*65}")
    print(f"ETAPA 2 — Playwright: {len(pendientes):,} páginas pendientes de detalle")
    print(f"(de {len(todos):,} totales — ya hechas o filtradas por idioma se saltan)")
    print(f"{'='*65}")

    if not pendientes:
        print("  ✅ Nada pendiente en Etapa 2.")
        return

    with sync_playwright() as p:
        browser, context, page = crear_browser(p)
        fallos_seguidos = 0

        try:
            for i, curso in enumerate(pendientes, 1):
                url = curso.get("url")
                if not url:
                    continue

                try:
                    detalle = extraer_detalle_curso(page, url)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # Acá caen: browser/context cerrado por suspensión del PC,
                    # timeouts, pérdida de red, etc. NO matamos el script.
                    fallos_seguidos += 1
                    print(f"  ⚠ [{i}/{len(pendientes)}] Error visitando {url}: {type(e).__name__}: {e}")

                    if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                        print(f"\n⛔ {fallos_seguidos} fallos seguidos — reiniciando navegador "
                              f"y esperando {ESPERA_TRAS_FALLO_S}s...")
                        guardar_csv(todos)
                        time.sleep(ESPERA_TRAS_FALLO_S)
                        try:
                            context.close()
                            browser.close()
                        except Exception:
                            pass
                        browser, context, page = crear_browser(p)
                        fallos_seguidos = 0
                    else:
                        time.sleep(random.uniform(5, 10))
                    continue  # este curso queda pendiente para la próxima corrida

                fallos_seguidos = 0

                url_real = detalle.pop("url_real", url)
                if url_real and url_real != url:
                    curso["url"] = url_real
                curso["tipo_url"] = detectar_tipo_url(url_real or url)

                # rating/num_reseñas/precio: la API ya no los trae (confirmado
                # con la respuesta cruda), así que usamos lo que sacó la
                # página si el campo sigue vacío.
                rating_pagina      = detalle.pop("rating_pagina", "")
                num_reseñas_pagina = detalle.pop("num_reseñas_pagina", "")
                precio_pagina      = detalle.pop("precio_pagina", "")
                if not (curso.get("rating") or "").strip() and rating_pagina:
                    curso["rating"] = rating_pagina
                if not (curso.get("num_reseñas") or "").strip() and num_reseñas_pagina:
                    curso["num_reseñas"] = num_reseñas_pagina
                if not (curso.get("precio") or "").strip() and precio_pagina:
                    curso["precio"] = precio_pagina

                curso.update(detalle)

                checks = {
                    "headline":    "✅" if detalle.get("headline")       else "—",  # legítimamente vacío casi siempre
                    "rating":      "✅" if curso.get("rating")            else "❌",
                    "horas":       "✅" if detalle.get("horas_semana") or detalle.get("duracion_total") else "❌",
                    "qué_aprend":  "✅" if detalle.get("que_aprenderas") else "❌",
                    "habilidades": "✅" if detalle.get("habilidades")    else "❌",
                    "temario":     "✅" if detalle.get("temario")        else "❌",
                }
                tipo_tag = f"[{curso['tipo_url'][:4]}]"
                print(
                    f"  [{i:>4}/{len(pendientes)}] {tipo_tag} {curso.get('titulo','')[:40]:40s} | "
                    + " ".join(f"{k}:{v}" for k, v in checks.items())
                )

                if i % 10 == 0:
                    guardar_csv(todos)
                    print(f"  💾 Backup Etapa 2: {i}/{len(pendientes)}")

                time.sleep(random.uniform(2.0, 4.0))

        except KeyboardInterrupt:
            print("\n\n⏸  Pausado por el usuario (Ctrl+C). Guardando progreso...")
        finally:
            guardar_csv(todos)
            try:
                browser.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    existente = cargar_csv_existente()

    todos = etapa1_api()
    if not todos:
        print("❌ No se obtuvieron cursos. Abortando.")
        return

    todos = merge_con_existente(todos, existente)
    guardar_csv(todos)
    print(f"\n✅ Etapa 1 completa (mergeada con progreso previo): {len(todos):,} cursos → {OUTPUT_CSV}")

    if SCRAPE_DETALLES:
        etapa2_playwright(todos)

    df = pd.DataFrame(todos)
    if len(df):
        df.drop_duplicates(subset=["id_coursera"], inplace=True)

    # Guardamos el CSV "crudo" completo (incluye proyectos y otros idiomas,
    # por si en algún momento los querés) y aparte el filtrado final.
    guardar_csv(df.to_dict("records"))

    if len(df):
        df_filtrado = df[
            df["tipo_url"].isin(["curso", "especializacion"])
            & df["idioma"].apply(idioma_permitido)
        ].copy()
        filtrado_csv = OUTPUT_CSV.replace(".csv", "_filtrado_en_es.csv")
        df_filtrado.to_csv(filtrado_csv, sep=";", index=False, encoding="utf-8-sig")
        print(f"\n📦 Dataset filtrado (solo curso/especializacion, en/es): "
              f"{len(df_filtrado):,} filas → {filtrado_csv}")

    print(f"\n{'='*65}")
    print(f"✅ COMPLETADO  |  Total crudo: {len(df):,}  |  Archivo: {OUTPUT_CSV}")
    print(f"{'='*65}")

    print(f"\n{'='*65}")
    print("COBERTURA DE CAMPOS (sobre el dataset crudo)")
    print(f"{'='*65}")
    for col, fuente in [
        ("rating","API+PW"), ("num_reseñas","API+PW"), ("nivel","API"),
        ("precio","API+PW"), ("idioma","API"), ("instructor","API"),
        ("universidad","API"), ("objetivos_api","API (la API ya no manda skills, casi siempre vacío)"),
        ("headline","PW (vacío salvo cursos parte de especialización)"),
        ("num_inscritos","PW (Coursera ya no lo muestra, siempre vacío)"),
        ("horas_semana","PW"), ("duracion_semanas","PW"),
        ("duracion_total","PW"), ("cronograma","PW"),
        ("que_aprenderas","PW"), ("habilidades","PW (reemplaza a objetivos_api)"),
        ("herramientas","PW"), ("requisitos","PW (muchos cursos no tienen esta sección)"),
        ("temario","PW"), ("temario_stats","PW"),
    ]:
        if col in df.columns:
            pct = (df[col].astype(str).str.strip()
                   .replace("", pd.NA).notna().sum() / max(len(df),1) * 100)
            print(f"  {col:22s} [{fuente}]: {pct:5.1f}%")


if __name__ == "__main__":
    main()