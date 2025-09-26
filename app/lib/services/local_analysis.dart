import 'dart:io';
import 'dart:async';
//import 'dart:ui' as ui;
//import 'package:flutter/services.dart';
//import 'package:image/image.dart' as img;
//import 'package:path_provider/path_provider.dart';

import '../models/report.dart';
//import '../services/yolo_v11_detector.dart';

class LocalAnalysis {
  /*late YoloV11Detector _detector;
  bool _initialized = false;

  /// Инициализация модели YOLOv11
  Future<void> init() async {
    if (_initialized) return;
    _detector = await YoloV11Detector.create(
      classNames: ["tree"], // ⚠️ Замени своими классами
      modelAsset: "assets/models/best_float32.tflite",
    );
    _initialized = true;
  }

  Future<Report> analyzeImage(File imageFile) async {
    if (!_initialized) {
      await init();
    }

    final bytes = await imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final ui.Image uiImage = frame.image;

    final detections = await _detector.detect(uiImage);

    final img.Image baseImage = img.decodeImage(bytes)!;

    // Рисуем все детекции
    for (final det in detections) {
      img.drawRect(
        baseImage,
        det.x.round(),
        det.y.round(),
        (det.x + det.width).round(),
        (det.y + det.height).round(),
        img.getColor(255, 0, 0),
      );

      img.drawString(
        baseImage,
        img.arial_24,
        det.x.round(),
        (det.y - 24).clamp(0, baseImage.height - 24).round(),
        "${det.className} ${(det.confidence * 100).toStringAsFixed(1)}%",
      );

    }

    final Directory dir = await getApplicationDocumentsDirectory();
    final String outPath =
        "${dir.path}/detected_${DateTime.now().millisecondsSinceEpoch}.png";

    final File outFile = File(outPath);
    await outFile.writeAsBytes(img.encodePng(baseImage));

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
  }*/
  Future<Report> analyzeImage(File imageFile) async {
   await Future.delayed(Duration(seconds: 2)); 
    return Report( 
      plantName: 'grass', 
      probability: 5.0, 
      species: 'Quercus robur', 
      trunkRot: 'Нет', 
      trunkHoles: 'Нет', 
      trunkCracks: 'Мелкие', 
      trunkDamage: 'Нет', 
      crownDamage: 'Лёгкие', 
      fruitingBodies: 'Есть', 
      diseases: 'Отсутствуют', 
      dryBranchPercentage: 10.0, 
      additionalInfo: 'Здоровое дерево', 
      imagePath: 'assets/images/wth_is_this.png',
      );
  }

}