# 🙈 Pixelar caras

Herramienta web para pixelar, difuminar o tapar caras en fotos. Todo se procesa **en el navegador**: las imágenes no se suben a ningún servidor.

## Funciones

- Detección automática de caras (modelo SSD MobileNet v1 de [face-api](https://github.com/vladmandic/face-api)).
- Modo «fotos de grupo»: analiza la imagen por zonas para encontrar caras pequeñas.
- Tres efectos: pixelado, difuminado y tapar en negro, con forma rectangular u ovalada.
- Retoque a mano: arrastra para tapar una cara no detectada y haz clic en un recuadro para activarlo o desactivarlo.
- Varias imágenes a la vez, que también se pueden pegar con Ctrl+V. Se descargan de una en una o todas juntas.
- Al exportar se eliminan los metadatos EXIF, incluida la ubicación GPS.

## Publicar en GitHub Pages

1. Crea un repositorio nuevo y sube **todo el contenido de esta carpeta**: `index.html`, `lib/` y `models/`.
2. Ve a **Settings → Pages**.
3. En *Source*, elige **Deploy from a branch**, rama `main` y carpeta `/ (root)`. Pulsa **Save**.
4. En uno o dos minutos estará disponible en `https://<tu-usuario>.github.io/<repositorio>/`.

## Probar en local

Si abres `index.html` con doble clic no funciona, porque el navegador bloquea la carga del modelo desde `file://`. Sírvelo con un servidor local:

```bash
python -m http.server 8000
# y abre http://localhost:8000
```

## Estructura

```
index.html                 interfaz y lógica
lib/face-api.js            librería de detección (incluye TensorFlow.js), licencia MIT
models/ssd_mobilenetv1_*   modelo de detección (~5,4 MB)
```

## Aviso

Ningún detector es perfecto. **Revisa siempre la imagen antes de publicarla**, sobre todo si salen menores.

## Versión de línea de comandos (Python)

La carpeta `python/` contiene `pixelar_caras.py`, una versión en Python con OpenCV que sirve para procesar carpetas enteras desde la terminal:

```bash
pip install opencv-python
python python/pixelar_caras.py fotos/ --bloques 8
```
