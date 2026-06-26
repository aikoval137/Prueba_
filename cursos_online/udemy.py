"""
====================================================================
UDEMY SCRAPER - VERSIÓN CON NAVEGADOR REAL (Playwright)
CON CHECKPOINT/RESUME + DETECCIÓN DE BLOQUEO
====================================================================
CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR:

  1) DETECCIÓN DE BLOQUEO EN ETAPA 2
     A partir de cierto volumen de requests Udemy empieza a redirigir
     las páginas de curso a contenido genérico (la barra de nav con
     "Explore by goal | Popular Issuers | Popular Subjects", o a un
     login). Antes esto se guardaba como si fuera un curso válido
     (con casi todo vacío). Ahora se detecta explícitamente, se
     reintenta con backoff, y si el bloqueo persiste el script se
     PAUSA solo (no sigue de cero) para que esperes o cambies de IP.

  2) CHECKPOINT / RESUME REAL
     - El CSV se lee al arrancar: si ya existe, los cursos ya
       encontrados en Etapa 1 NO se vuelven a buscar (se cargan tal
       cual) y los que ya tienen "temario" no vacío en Etapa 2 se
       saltan.
     - Podés parar el script en cualquier momento (Ctrl+C) o que se
       corte la luz: al volver a ejecutar, sigue donde quedó.
     - Etapa 1 también guarda un archivo de progreso por término
       (qué páginas ya recorrió de cada término) para no repetir
       búsquedas ya hechas.

INSTALACIÓN (una sola vez):
  pip install playwright pandas
  playwright install chromium

CÓMO EJECUTAR:
  python udemy_scraper_playwright.py
  (Ctrl+C para pausar en cualquier momento; volver a correr para continuar)
====================================================================
"""

import time
import random
import re
import json
import sys
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
OUTPUT_CSV     = "udemy_cursos_playwright_3.csv"
PROGRESS_FILE  = "udemy_progress.json"   # guarda qué páginas/términos ya se hicieron
MAX_PAGINAS    = 20
HEADLESS       = True

SCRAPE_DETALLES   = True
LIMITE_DETALLES   = None

# Si N páginas de curso seguidas vienen "bloqueadas" (sin contenido real),
# el script pausa y espera antes de seguir reintentando.
MAX_FALLOS_SEGUIDOS   = 5
ESPERA_TRAS_BLOQUEO_S = 120   # segundos que espera tras detectar bloqueo sostenido

TERMINOS = [
    "python", "javascript", "excel", "photoshop", "diseño grafico",
    "marketing digital", "autocad", "sql", "machine learning",
    "finanzas", "ingles", "wordpress", "video edicion", "illustrator",
    "react", "powerpoint", "arduino", "unity", "contabilidad", "seo",
]

BASE_URL = "https://www.udemy.com/courses/search/?q={termino}&p={pagina}"


# ─── UTILIDADES DE PERSISTENCIA ───────────────────────────────────
def cargar_csv_existente() -> pd.DataFrame:
    path = Path(OUTPUT_CSV)
    if path.exists():
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            print(f"📂 CSV existente encontrado: {len(df):,} cursos cargados (retomando)")
            return df
        except Exception as e:
            print(f"⚠ No se pudo leer el CSV existente ({e}), arrancando de cero")
    return pd.DataFrame()


def cargar_progreso() -> dict:
    path = Path(PROGRESS_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"terminos_completados": []}


def guardar_progreso(progreso: dict):
    Path(PROGRESS_FILE).write_text(json.dumps(progreso, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar_csv(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, sep=";", index=False, encoding="utf-8-sig")


def limpiar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)
    return " ".join(texto.split())


# ─── DETECCIÓN DE BLOQUEO ─────────────────────────────────────────
# Frases que solo aparecen en el menú de navegación genérico de Udemy
# (home / sidebar de "explorar categorías"), nunca en el temario real
# de un curso. Si aparecen, NO es la página del curso.
NAV_GENERICA_MARCADORES = [
    "explore by goal",
    "popular issuers",
    "popular subjects",
    "explore by",
    "career paths",
    "sign up",
    "log in",
]


def _es_nav_generica(texto: str) -> bool:
    t = (texto or "").lower()
    if not t.strip():
        return False
    return any(marca in t for marca in NAV_GENERICA_MARCADORES)


def detalle_parece_bloqueado(detalle: dict) -> bool:
    """
    Heurística: tratamos el detalle como bloqueado si:
      (a) descripción, objetivos y temario vienen todos vacíos, o
      (b) cualquiera de esos campos contiene texto de la navegación
          genérica de Udemy (ej: "Explore by goal | Popular Issuers |
          Popular Subjects"), que es lo que aparece cuando Udemy
          redirige a la home/login en vez de mostrar el curso.
    """
    campos_clave = ["descripcion", "objetivos", "temario"]
    valores = [detalle.get(c) or "" for c in campos_clave]

    todos_vacios = all(not v.strip() for v in valores)
    tiene_nav_generica = any(_es_nav_generica(v) for v in valores)

    return todos_vacios or tiene_nav_generica


def extraer_detalle_curso(page, url: str) -> dict:
    vacio = {"idioma": "", "descripcion": "", "objetivos": "", "requisitos": "", "temario": "", "temario_stats": ""}
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(random.uniform(1500, 2500))
    except Exception:
        return vacio

    script = """
    () => {
        const idioma = document.querySelector('[data-purpose="course-language"] span')?.textContent.trim() || '';
        const objetivos = Array.from(document.querySelectorAll('[data-purpose="objective"]'))
            .map(el => el.textContent.trim());

        let requisitos = [];
        const reqTitle = document.querySelector('h2[data-purpose="requirements-title"]');
        if (reqTitle) {
            const ul = reqTitle.parentElement.querySelector('ul');
            if (ul) requisitos = Array.from(ul.querySelectorAll('li')).map(li => li.textContent.trim());
        }

        const descEl = document.querySelector('[data-testid="safely-set-inner-html:description:description"]');
        const descripcion = descEl ? descEl.textContent.trim() : '';

        const navGenericaMarcadores = ['explore by goal', 'popular issuers', 'popular subjects', 'career paths'];
        const secciones = Array.from(
            document.querySelectorAll('[class*="section-title"]')
        )
            .filter(el => !el.className.includes('container') && !el.className.includes('stats'))
            .map(el => el.textContent.trim())
            .filter(t => !navGenericaMarcadores.some(m => t.toLowerCase().includes(m)));

        const statsEl = document.querySelector('[data-testid="curriculum-stats"]');
        const temarioStats = statsEl ? statsEl.textContent.trim() : '';

        return {
            idioma,
            objetivos: objetivos.join(' | '),
            requisitos: requisitos.join(' | '),
            descripcion,
            temario: secciones.join(' | '),
            temario_stats: temarioStats,
        };
    }
    """
    try:
        return page.evaluate(script)
    except Exception as e:
        print(f"    ⚠ Error extrayendo detalle: {e}")
        return vacio


def extraer_cursos_de_pagina(page) -> list[dict]:
    script = """
    () => {
        const tarjetas = document.querySelectorAll('section[class*="course-product-card--card"]');
        const resultados = [];
        tarjetas.forEach(card => {
            const get = (sel) => {
                const el = card.querySelector(sel);
                return el ? el.textContent.trim() : null;
            };
            const link = card.querySelector('a[href*="/course/"]');
            const tagTexts = Array.from(card.querySelectorAll('.tag-module--tag--4CWOQ, [class*="tag-module--tag"]'))
                .map(el => el.textContent.trim());
            const buscarTag = (...palabras) =>
                tagTexts.find(t => palabras.some(p => t.toLowerCase().includes(p))) || null;

            resultados.push({
                titulo: get('[class*="card-title-module--clipped"]') || (link ? link.textContent.trim() : null),
                headline: get('[data-testid*="course-headline"]'),
                instructor: get('[data-testid*="visible-instructors"]'),
                rating: get('[data-purpose="rating-number"]'),
                num_reseñas: buscarTag('valoracion', 'rating'),
                horas: buscarTag('hora'),
                num_clases: buscarTag('clase'),
                nivel: buscarTag('nivel', 'principiante', 'intermedio', 'experto', 'avanzado'),
                precio: get('[data-purpose="course-price-text"]'),
                url: link ? link.href : null,
            });
        });
        return resultados;
    }
    """
    try:
        return page.evaluate(script)
    except Exception as e:
        print(f"  ⚠ Error extrayendo datos de la página: {e}")
        return []


def scrape_termino(browser_page, termino: str, vistos: set) -> list[dict]:
    cursos_nuevos = []
    print(f"\n🔍 Término: '{termino}'")

    for pagina in range(1, MAX_PAGINAS + 1):
        url = BASE_URL.format(termino=termino.replace(" ", "+"), pagina=pagina)
        try:
            browser_page.goto(url, wait_until="domcontentloaded", timeout=20000)
            time.sleep(random.uniform(2.0, 3.5))
            cursos_pagina = extraer_cursos_de_pagina(browser_page)

            if not cursos_pagina:
                print(f"  📄 Página {pagina}: 0 cursos. Fin del término.")
                break

            nuevos_esta_pagina = 0
            for item in cursos_pagina:
                clave = item.get("url") or item.get("titulo")
                if not clave or clave in vistos:
                    continue
                vistos.add(clave)
                nuevos_esta_pagina += 1
                cursos_nuevos.append({
                    "titulo": limpiar_texto(item.get("titulo") or ""),
                    "headline": limpiar_texto(item.get("headline") or ""),
                    "instructor": limpiar_texto(item.get("instructor") or ""),
                    "rating": item.get("rating"),
                    "num_reseñas": limpiar_texto(item.get("num_reseñas") or ""),
                    "horas": limpiar_texto(item.get("horas") or ""),
                    "num_clases": limpiar_texto(item.get("num_clases") or ""),
                    "nivel": limpiar_texto(item.get("nivel") or ""),
                    "precio": limpiar_texto(item.get("precio") or ""),
                    "url": item.get("url"),
                    "termino_busqueda": termino,
                    # columnas de detalle, se llenan en Etapa 2
                    "idioma": "", "descripcion": "", "objetivos": "",
                    "requisitos": "", "temario": "", "temario_stats": "",
                })

            print(f"  📄 Página {pagina}: {len(cursos_pagina)} cursos "
                  f"({nuevos_esta_pagina} nuevos, total: {len(cursos_nuevos):,})")

            if nuevos_esta_pagina == 0:
                print(f"  🏁 Sin cursos nuevos, siguiente término")
                break

            time.sleep(random.uniform(1.5, 3.0))

        except Exception as e:
            print(f"  ❌ Error en página {pagina}: {e}")
            break

    return cursos_nuevos


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


def main():
    df_existente = cargar_csv_existente()
    progreso = cargar_progreso()

    todos = df_existente.to_dict("records") if len(df_existente) else []
    # aseguramos que existan las columnas de detalle aunque el CSV viejo no las tuviera
    for c in ["idioma", "descripcion", "objetivos", "requisitos", "temario", "temario_stats"]:
        for curso in todos:
            curso.setdefault(c, "")

    vistos = {c["url"] for c in todos if c.get("url")}

    print("=" * 65)
    print("UDEMY SCRAPER - NAVEGADOR REAL (Playwright) - con checkpoint")
    print(f"Cursos ya en CSV: {len(todos):,} | Términos completados antes: "
          f"{len(progreso['terminos_completados'])}")
    print("=" * 65)

    with sync_playwright() as p:
        browser, context, page = crear_browser(p)

        try:
            # ── ETAPA 1: búsqueda por término ──────────────────────
            for termino in TERMINOS:
                if termino in progreso["terminos_completados"]:
                    print(f"⏭  Saltando '{termino}' (ya completado en una corrida anterior)")
                    continue

                nuevos = scrape_termino(page, termino, vistos)
                todos.extend(nuevos)

                if todos:
                    guardar_csv(pd.DataFrame(todos))
                    print(f"  💾 Backup guardado: {len(todos):,} cursos totales")

                progreso["terminos_completados"].append(termino)
                guardar_progreso(progreso)

                time.sleep(random.uniform(3, 6))

            # ── ETAPA 2: detalle de cada curso ─────────────────────
            if SCRAPE_DETALLES and todos:
                pendientes = [c for c in todos if not (c.get("temario") or "").strip()]
                if LIMITE_DETALLES is not None:
                    pendientes = pendientes[:LIMITE_DETALLES]

                print(f"\n{'='*65}")
                print(f"ETAPA 2: {len(pendientes):,} cursos pendientes de detalle "
                      f"(de {len(todos):,} totales — los que ya tienen temario se saltan)")
                print(f"{'='*65}")

                fallos_seguidos = 0

                for i, curso in enumerate(pendientes, 1):
                    url = curso.get("url")
                    if not url:
                        continue

                    detalle = extraer_detalle_curso(page, url)

                    if detalle_parece_bloqueado(detalle):
                        fallos_seguidos += 1
                        print(f"  🚫 [{i}/{len(pendientes)}] Posible bloqueo/redirect en: {url}")

                        if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                            print(f"\n⛔ {fallos_seguidos} fallos seguidos — probablemente Udemy "
                                  f"está bloqueando la sesión.")
                            print(f"   Guardando progreso y esperando {ESPERA_TRAS_BLOQUEO_S}s, "
                                  f"luego reinicio el navegador (nueva sesión)...")
                            guardar_csv(pd.DataFrame(todos))
                            time.sleep(ESPERA_TRAS_BLOQUEO_S)
                            try:
                                context.close()
                                browser.close()
                            except Exception:
                                pass
                            browser, context, page = crear_browser(p)
                            fallos_seguidos = 0
                        else:
                            time.sleep(random.uniform(5, 10))  # backoff antes de seguir
                        continue  # no actualizamos el curso con datos vacíos; queda "pendiente" para el próximo run

                    fallos_seguidos = 0
                    curso.update(detalle)
                    print(f"  [{i}/{len(pendientes)}] {curso.get('titulo', '')[:50]:50s} "
                          f"| idioma: {detalle.get('idioma','')} | temario: ✅")

                    if i % 10 == 0:
                        guardar_csv(pd.DataFrame(todos))
                        print(f"  💾 Backup guardado (etapa 2): {i}/{len(pendientes)}")

                    time.sleep(random.uniform(2.0, 4.0))

        except KeyboardInterrupt:
            print("\n\n⏸  Pausado por el usuario (Ctrl+C). Guardando progreso...")
        finally:
            guardar_csv(pd.DataFrame(todos))
            guardar_progreso(progreso)
            browser.close()

    df = pd.DataFrame(todos)
    if len(df):
        df.drop_duplicates(subset=["url"], inplace=True)
    guardar_csv(df)

    print(f"\n{'='*65}")
    print(f"✅ Estado actual guardado")
    print(f"   Cursos únicos:  {len(df):,}")
    print(f"   Con temario:    {(df['temario'].astype(str).str.strip() != '').sum():,}")
    print(f"   Archivo:        {OUTPUT_CSV}")
    print(f"{'='*65}")
    print("Si quedaron cursos sin temario (bloqueo), volvé a correr el script: "
          "solo reintentará los pendientes.")


if __name__ == "__main__":
    main()