import 'dart:io';
import 'dart:async';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../models/report.dart';
import '../services/yolo_v11_detector.dart';

class LocalAnalysis {
  late YoloV11Detector _detector;
  bool _initialized = false;

  /// Инициализация модели YOLOv11
  Future<void> init() async {
    if (_initialized) return;
    _detector = await YoloV11Detector.create(
      classNames: ["tree"], // ⚠️ Замени своими классами
      modelAsset: "assets/best_float32.tflite",
    );
    _initialized = true;
  }

  Future<Report> analyzeImage(File imageFile) async {
    if (!_initialized) {
      await init();
    }

    // Загружаем bytes
    final bytes = await imageFile.readAsBytes();

    // Получаем ui.Image (для модели)
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final ui.Image uiImage = frame.image;

    // Детекция
    final detections = await _detector.detect(uiImage);
    print('DEBUG: detections count=${detections.length}');
    for (var det in detections) {
      print('DEBUG: ${det.className} x=${det.x} y=${det.y} w=${det.width} h=${det.height} conf=${det.confidence}');
    }


    // Преобразуем ui.Image → img.Image (RGBA) для рисования
    final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      print('ERROR: Failed to convert ui.Image to bytes');
      return Report(plantName: 'Ошибка', imagePath: '');
    }
    final Uint8List rgba = byteData.buffer.asUint8List();
    final img.Image baseImage = img.Image.fromBytes(
      uiImage.width,
      uiImage.height,
      rgba,
      format: img.Format.rgba,
    );
    print('DEBUG: baseImage w=${baseImage.width}, h=${baseImage.height}');
    

 // width = 100-10, height = 100-10




    // Рисуем все детекции
    for (final det in detections) {
      img.drawRect(
        baseImage,
        det.x.round(),
        det.y.round(),
        (det.x + det.width).round(),
        (det.y + det.height).round(),
        img.getColor(255, 0, 0), // ярко-красный
      );

      img.drawString(
        baseImage,
        img.arial_24,
        det.x.round(),
        (det.y - 24).clamp(0, baseImage.height - 24).round(),
        "${det.className} ${(det.confidence * 100).toStringAsFixed(1)}%",
        color: img.getColor(255, 255, 0), // жёлтая подпись
      );
    }

    // Сохраняем изображение с детекциями
    final Directory dir = await getApplicationDocumentsDirectory();
    final String outPath =
        "${dir.path}/detected_${DateTime.now().millisecondsSinceEpoch}.png";
    final File outFile = File(outPath);
    await outFile.writeAsBytes(img.encodePng(baseImage));
    print('DEBUG: saved output to ${outFile.path}, exists=${outFile.existsSync()}');

    // Формируем отчёт
    if (detections.isEmpty) {
      return Report(
        plantName: "Не найдено",
        imagePath: outPath,
      );
    }

    final first = detections.first;
    return Report(
      plantName: first.className,
      probability: first.confidence,
      imagePath: outPath,
    );
  }




}