#!/usr/bin/env python3
"""
pixelar_caras.py — Detecta caras en imágenes y las pixela (o difumina).

Uso:
    python pixelar_caras.py foto.jpg
    python pixelar_caras.py carpeta_fotos/ -o salida/
    python pixelar_caras.py foto.jpg --bloques 8 --margen 0.3
    python pixelar_caras.py foto.jpg --modo blur
    python pixelar_caras.py foto.jpg --marcar      # solo dibuja recuadros (para revisar)

Requisitos:
    pip install opencv-python

Detector:
    - YuNet (red neuronal, muy preciso). Se descarga solo la primera vez
      (~230 KB) junto al script.
    - Si no hay internet ni modelo, usa Haar Cascade (incluido en OpenCV,
      menos preciso, sobre todo con caras de perfil o pequeñas).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------- detectores
def obtener_modelo_yunet():
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000:
        return YUNET_PATH
    try:
        print("Descargando modelo YuNet (solo la primera vez)...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        if YUNET_PATH.stat().st_size > 100_000:
            return YUNET_PATH
    except Exception as e:
        print(f"  No se pudo descargar ({e}).")
    YUNET_PATH.unlink(missing_ok=True)
    return None


class DetectorYuNet:
    nombre = "YuNet"

    def __init__(self, modelo, confianza):
        self.det = cv2.FaceDetectorYN.create(str(modelo), "", (320, 320),
                                             confianza, 0.3, 5000)

    def detectar(self, img):
        h, w = img.shape[:2]
        # Reducimos imágenes enormes para ir rápido, y luego reescalamos cajas
        escala = min(1.0, 1600 / max(h, w))
        peque = cv2.resize(img, None, fx=escala, fy=escala) if escala < 1 else img
        ph, pw = peque.shape[:2]
        self.det.setInputSize((pw, ph))
        _, caras = self.det.detect(peque)
        if caras is None:
            return []
        return [tuple(int(v / escala) for v in c[:4]) for c in caras]


class DetectorHaar:
    nombre = "Haar Cascade"

    def __init__(self):
        base = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.perfil = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.frontal.empty() or self.perfil.empty():
            sys.exit("Tu versión de OpenCV no incluye los ficheros Haar y no hay modelo YuNet.\n"
                     f"Descarga el modelo a mano y ponlo junto al script:\n  {YUNET_URL}")

    def detectar(self, img):
        gris = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        w = gris.shape[1]
        cajas = list(self.frontal.detectMultiScale(gris, 1.1, 5, minSize=(24, 24)))
        cajas += list(self.perfil.detectMultiScale(gris, 1.1, 5, minSize=(24, 24)))
        # Perfil mirando al otro lado: volteamos la imagen
        for (x, y, cw, ch) in self.perfil.detectMultiScale(cv2.flip(gris, 1), 1.1, 5,
                                                           minSize=(24, 24)):
            cajas.append((w - x - cw, y, cw, ch))
        if not cajas:
            return []
        # Fusiona detecciones solapadas
        rects, _ = cv2.groupRectangles([list(map(int, c)) for c in cajas] * 2, 1, 0.3)
        return [tuple(r) for r in rects]


# ------------------------------------------------------------- anonimizado
def ampliar(caja, margen, w, h):
    x, y, cw, ch = caja
    dx, dy = int(cw * margen), int(ch * margen)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(w, x + cw + dx), min(h, y + ch + dy)
    return x1, y1, x2, y2


def pixelar(region, bloques):
    rh, rw = region.shape[:2]
    bw, bh = max(1, bloques), max(1, round(bloques * rh / rw))
    peque = cv2.resize(region, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(peque, (rw, rh), interpolation=cv2.INTER_NEAREST)


def difuminar(region):
    rh, rw = region.shape[:2]
    k = max(rw, rh) // 2 * 2 + 1  # kernel impar, proporcional al tamaño
    return cv2.GaussianBlur(region, (k, k), 0)


def procesar(img, detector, args):
    h, w = img.shape[:2]
    caras = detector.detectar(img)
    salida = img.copy()
    for caja in caras:
        x1, y1, x2, y2 = ampliar(caja, args.margen, w, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        if args.marcar:
            cv2.rectangle(salida, (x1, y1), (x2, y2), (0, 0, 255), 3)
            continue
        region = salida[y1:y2, x1:x2]
        nueva = pixelar(region, args.bloques) if args.modo == "pixel" else difuminar(region)
        if args.ovalo:
            mascara = np.zeros(region.shape[:2], np.uint8)
            cv2.ellipse(mascara, ((x2 - x1) // 2, (y2 - y1) // 2),
                        ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
            nueva = np.where(mascara[..., None] == 255, nueva, region)
        salida[y1:y2, x1:x2] = nueva
    return salida, len(caras)


# ------------------------------------------------------------------- main
def leer(ruta):
    # imdecode permite rutas con tildes/ñ en Windows
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar(ruta, img):
    ok, buf = cv2.imencode(ruta.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(ruta))
    return ok


def main():
    p = argparse.ArgumentParser(description="Pixela las caras de una imagen o carpeta.")
    p.add_argument("entrada", help="Imagen o carpeta de imágenes")
    p.add_argument("-o", "--salida", help="Archivo o carpeta de salida "
                   "(por defecto: <nombre>_pixelada.<ext> / carpeta <entrada>_pixeladas)")
    p.add_argument("--modo", choices=["pixel", "blur"], default="pixel",
                   help="pixel (por defecto) o blur (difuminado)")
    p.add_argument("--bloques", type=int, default=10,
                   help="Nº de bloques a lo ancho de cada cara. Menos = más pixelado (def. 10)")
    p.add_argument("--margen", type=float, default=0.25,
                   help="Margen extra alrededor de la cara, en proporción (def. 0.25)")
    p.add_argument("--confianza", type=float, default=0.6,
                   help="Umbral de YuNet 0-1. Bájalo si se escapan caras (def. 0.6)")
    p.add_argument("--ovalo", action="store_true", help="Forma ovalada en vez de rectángulo")
    p.add_argument("--marcar", action="store_true",
                   help="No pixela: dibuja recuadros rojos para revisar la detección")
    p.add_argument("--haar", action="store_true", help="Forzar el detector Haar (sin red)")
    args = p.parse_args()

    entrada = Path(args.entrada)
    if not entrada.exists():
        sys.exit(f"No existe: {entrada}")

    modelo = None if args.haar else obtener_modelo_yunet()
    detector = DetectorYuNet(modelo, args.confianza) if modelo else DetectorHaar()
    print(f"Detector: {detector.nombre}")

    if entrada.is_dir():
        archivos = sorted(f for f in entrada.iterdir() if f.suffix.lower() in EXTENSIONES)
        destino = Path(args.salida) if args.salida else entrada.with_name(entrada.name + "_pixeladas")
        destino.mkdir(parents=True, exist_ok=True)
        pares = [(f, destino / f.name) for f in archivos]
    else:
        destino = Path(args.salida) if args.salida else entrada.with_name(
            f"{entrada.stem}_pixelada{entrada.suffix}")
        pares = [(entrada, destino)]

    total = 0
    for origen, dest in pares:
        img = leer(origen)
        if img is None:
            print(f"  ✗ {origen.name}: no se pudo leer")
            continue
        resultado, n = procesar(img, detector, args)
        total += n
        guardar(dest, resultado)
        aviso = "  ⚠ ninguna cara detectada" if n == 0 else ""
        print(f"  ✓ {origen.name}: {n} cara(s) → {dest}{aviso}")

    print(f"Hecho. {len(pares)} imagen(es), {total} cara(s).")


if __name__ == "__main__":
    main()
