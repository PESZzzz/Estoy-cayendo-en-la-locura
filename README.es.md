# PESZzzz-modly-hunyuan3d-2.1 :D!

[English](README.md) | **Español**

## ¿Qué es esto?

Prácticamente, una extensión de Modly para Hunyuan3D 2.1 diseñada para usuarios de AMD en Windows y laptops o PCs modestas.

Reemplaza y parcha el código interno de Hunyuan3D 2.1 para ejecutarlo en hardware para el cual Tencent nunca optimizó[cite: 3].

Como dije, esto está diseñado para Modly. **NO** es un script independiente. Pero, hey, toda la estructura dentro de la carpeta `hy3dshape` contiene los parches. Si quieres conectarlo a ComfyUI o a tu propia herramienta, ¡adelante! (Igual no recomiendo ComfyUI para nada, especialmente para AMD).

En las pruebas demostró que funciona en varias computadoras, incluso con NVIDIA. Aunque también puede *explotar*, así que... cuidado (take care).

Oye, ¿sigues aquí? Genial, ¡eso significa que sabes leer :D!

---

## ¿Qué cambié?

Sinceramente, *no* me acuerdo. **NO** soy programador :D! Solo... usé IA.

Pero no te preocupes, encontrarás todas las explicaciones y cambios justo aquí:  
👉 **[Lee el desglose técnico (summary.md)](summary.md)**

Los archivos principales parchados son estos:
- `pipelines.py`
- `surface_extractors.py`
- `volume_decoders.py`

---

## Requisitos

Llegó la *MEJOR* parte:

| Componente   | Mínimo                | Recomendado               |
| ------------ | --------------------- | ------------------------- |
| **SO**       | Windows 10/11, Linux  | Windows 11 / Ubuntu 22.04 |
| **RAM**      | **16 GB**             | 32 GB                     |
| **GPU**      | No requerida          | AMD / NVIDIA / Intel      |
| **CPU**      | AMD Ryzen 5           | AMD Ryzen 7               |
| **Espacio**  | 15 GB libres          | 20 GB libres              |
| **Modly**    | Última versión estable| Última versión estable    |

> 📌 **Nota:** Esta extensión fue hecha pensando en usuarios de **AMD**. Si tienes una **GPU NVIDIA**, tal vez prefieras la extensión de AlefK1708: [modly-hunyuan3d-21-lowvram](https://github.com/Alefk1708/modly-hunyuan3d-21-lowvram).

---

## Instalación

1. Abre Modly.
2. Ve a **Extensions** (Extensiones).
3. Haz clic en **Install from GitHub** (Instalar desde GitHub).
4. Pega el enlace del repositorio: `https://github.com/PESZzzz/PESZzzz-modly-hunyuan3d-2.1`

¡Así de fácil!

---

## Instalación Manual (si sabes lo que haces)

1. En GitHub, presiona el botón verde **Code** y dale a **Download ZIP**.

...

¿Esperabas algo "técnico" como `git clone`? Solo puedes... ya sabes, descargarlo.

---

## Uso en Modly

### Parámetros Recomendados

| Parámetro            | Rápido      | Balanceado | Alta Calidad |
| -------------------- | ----------- | ---------- | ------------ |
| **Quality**          | 15 pasos    | 30 pasos   | 50 pasos     |
| **Mesh Resolution**  | 256         | 384        | 512          |
| **Guidance Scale**   | \-          | Cualquier escala | \-     |
| **Max Vertices**     | \-          | Depende del modelo | \-   |

**En más detalle:**
- **Quality (Calidad):** Determina los pasos de difusión.
- **Mesh Resolution (Resolución de Malla):** 256 es seguro para 16 GB de RAM. 384+ requiere más RAM / VRAM.
- **Guidance Scale:** Qué tan fielmente sigue la malla a la imagen de entrada.
- **Max Vertices:** Reduce la cantidad de polígonos al límite que le pongas.
- **Seed (Semilla):** Déjala en `-1` para que sea aleatoria. Cámbiala solo si sabes lo que haces.

---

## ¡¡¡IMPORTANTE!!!

1. **La generación es muy lenta en CPU.**  
   ¿Qué esperabas? En una laptop de 8 núcleos usando 30 pasos, puede tardar de 4 a 5 horas o más. Te recomiendo *MUCHO* usar 15 pasos y una resolución de malla baja para vistas previas rápidas.

2. **Cierra TODAS las aplicaciones que tengas abiertas.**  
   Los primeros minutos de carga consumen un montón de RAM. Conforme pasa el tiempo después de cargar, se estabiliza. ¡Hasta puedes ver videos en YouTube mientras esperas!

3. **Este proyecto sigue en desarrollo.**  
   Si encuentras algún problema, avísame en Twitter/X: [@SinJeshua](https://x.com/SinJeshua).

---

Finalmente, este proyecto tomó muchísimo trabajo. No soy programador; solo soy otro estudiante con una laptop normal, sin experiencia ni dinero. Todas estas ideas de optimización, como la "Estrategia Híbrida", las hice yo, no la IA. Esto fue toda una experiencia de aprendizaje para mí y aprendí un montón durante el proceso.

Si quieres apoyarme, puedes invitarme a un café en Ko-fi:  
☕ **[https://ko-fi.com/peszs](https://ko-fi.com/peszs)**

¡Cualquier apoyo me ayuda un montón!

*Dato curioso: Las herramientas de IA que usé para hacer este proyecto me hicieron llorar. A veces son tan tontas D:*

---

## Preguntas Frecuentes (FAQ)

**P: Tengo 16 GB de RAM, ¿mi PC va a colapsar durante la carga?**  
R: ¡Para nada :D! Como dije: cierra **todas** las apps que estés usando. Los primeros minutos usan mucha RAM, pero luego de cargar se estabiliza. ¡Hasta puedes ver YouTube mientras esperas!

**P: ¿Cuál es la configuración más rápida para probar si funciona en mi equipo?**  
R: 15 pasos, 256 de Mesh Resolution y 3 o 4 de Guidance Scale. Prueba primero con una imagen sencilla.

**P: ¿Por qué dices que puede "explotar" en tarjetas NVIDIA?**  
R: Tampoco es que vaya a explotar *literalmente*... ¡o tal vez sí :D! Pero en serio, este mod fue diseñado para AMD. Si tienes NVIDIA, el mod de AlefK1708 está muchísimo mejor optimizado para CUDA.

**P: ¿Puedo usar este código fuera de Modly?**  
R: ¡Totalmente! La carpeta `hy3dshape` contiene todos los parches personalizados.

**P: ¿Qué se supone que haga si no te acuerdas de lo que cambiaste?**  
R: ¡No te preocupes! Revisa el archivo **[summary.md](summary.md)** o busca dentro del código los comentarios etiquetados con `[COMMUNITY]`.

**P: ¿Vas a actualizarlo?**  
R: Tal vez. No esperes actualizaciones o parches seguidos. ¡Solo soy un estudiante!

**P: ¿Se puede optimizar aún más?**  
R: Definitivamente sí. Tal vez pueda meterle más optimizaciones en el futuro.

---

## Licencias y Avisos Legales

Este proyecto combina, modifica y extiende código de tecnologías de código abierto sujetas a sus respectivas licencias:

* **Tencent Hunyuan 3D 2.1:** Distribuido bajo el **TENCENT HUNYUAN 3D 2.1 COMMUNITY LICENSE AGREEMENT**. Por favor revisa el archivo [`LICENSE`](LICENSE) para ver los términos legales completos y la Política de Uso Aceptable (Exhibición A).
* **Base de Extensión para Modly:** Distribuido bajo la **Licencia MIT** Copyright (c) 2026 Lightning Pixel.
* **Componentes y Dependencias de Terceros:** Esta distribución incorpora componentes y conceptos de trabajos de terceros, incluyendo **Stable Diffusion** (Stability AI) y **Flux** (Black Forest Labs).

> **Aviso Importante:** En cumplimiento con las licencias originales, todos los avisos de terceros, créditos y licencias de código abierto se conservan en el archivo [`NOTICE`](NOTICE) ubicado en el directorio raíz.

Los parches de la comunidad y las modificaciones introducidas en este repositorio se lanzan bajo los mismos términos del proyecto original.

[![Licencia: Tencent](https://img.shields.io/badge/licencia-Tencent%20Hunyuan%20Community%20License-blue.svg)](LICENSE)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
