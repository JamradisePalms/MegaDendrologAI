import 'dart:io';
import 'dart:async';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../models/report.dart';
import '../services/yolo_v11_detector.dart';
import 'dart:math';
import '../services/resnet_classifier.dart';
import 'package:flutter/foundation.dart';
import 'package:intl/intl.dart'; // для форматирования даты


class LocalAnalysis {
  late YoloV11Detector _detector;
  late ResnetClassifier _resnetClassifier; // поле класса
  bool _initialized = false;

  /// Инициализация модели YOLOv11 + ResNet
  Future<void> init() async {
    if (_initialized) return;

    final List<String> treeClassNames = [
      'He определено', 'Береза', 'Боярышник', 'Вяз', 'Дерен белый',
      'Дуб', 'Ель', 'Ива', 'Карагана древовидная', 'Кизильник',
      'Клен остролистный', 'Клен ясенелистный', 'Лапчатка кустарниковая',
      'Лещина', 'Липа', 'Лиственница', 'Осина', 'Пузыреплодник калинолистный',
      'Роза морщинистая', 'Роза собачья', 'Рябина', 'Сирень обыкновенная',
      'Сосна', 'Спирея', 'Туя', 'Чубушник', 'Ясень'
    ];

    _detector = await YoloV11Detector.create(
      classNames: ["tree", "bush"], // ⚠️ Замени своими классами
      modelAsset: "assets/models/best_float32.tflite",
    );

    _resnetClassifier = ResnetClassifier();
    await _resnetClassifier.init(
      modelAsset: 'assets/models/simple_model_float32.tflite',
      classNames: treeClassNames,
    );

    _initialized = true;
  }

  Future<List<Report>> analyzeImage(File imageFile) async {
    if (!_initialized) {
      await init();
    }

    final bytes = await imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final ui.Image uiImage = frame.image;

    final detections = await _detector.detect(uiImage);
    debugPrint('DEBUG: detections count=${detections.length}');

    final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      return [Report(plantName: 'Ошибка', imagePath: '')];
    }
    final Uint8List rgba = byteData.buffer.asUint8List();
    final img.Image baseImage = img.Image.fromBytes(
      uiImage.width,
      uiImage.height,
      rgba,
      format: img.Format.rgba,
    );

    final Directory dir = await getApplicationDocumentsDirectory();
    final appDir = Directory('${dir.path}/PlantGuard');
    if (!await appDir.exists()) {
      await appDir.create(recursive: true);
    }
    final List<Report> reports = [];

    for (int i = 0; i < detections.length; i++) {
      final det = detections[i];

      // --- вырезаем по координатам ---
      final int x = det.x.round().clamp(0, baseImage.width - 1);
      final int y = det.y.round().clamp(0, baseImage.height - 1);
      final int w = det.width.round().clamp(1, baseImage.width - x);
      final int h = det.height.round().clamp(1, baseImage.height - y);

      final img.Image cropped = img.copyCrop(baseImage, x, y, w, h);

      // --- классификация через ResNet ---
      String label = 'Не определено';
      double probability = 0.0;
      String top3species = 'Не определено';
      try {
        final result = await _resnetClassifier.classify(cropped);

        // сортируем по вероятности
        final sorted = result.entries.toList()
          ..sort((a, b) => b.value.compareTo(a.value));

        // берем лучший
        final best = sorted.first;
        label = best.key;
        probability = best.value * 100;

        // берём топ-3 и делаем красивую строку "Берёза (75%), Дуб (15%), Ясень (10%)"
        final top3 = sorted.take(3).map((e) =>
            "${e.key} ${(e.value * 100).toStringAsFixed(1)}%").join(", ");
        top3species = top3;

      } catch (e) {
        debugPrint('ResNet classification error: $e');
      }

      // --- сохраняем вырезанное растение ---
      final String outPath =
          "${appDir.path}/plant_${DateTime.now().millisecondsSinceEpoch}_$i.png";
      final File outFile = File(outPath);
      await outFile.writeAsBytes(img.encodePng(cropped));

      final now = DateTime.now();
      final formattedDate = DateFormat('yyyy-MM-dd HH:mm').format(now);

      reports.add(
        Report(
          plantName: "$label $formattedDate",   // лучший класс + дата
          probability: double.parse((det.confidence * 100).toStringAsFixed(2)),
          species: top3species,                 // теперь тут топ-3
          imagePath: outFile.path,
        ),
      );


    }

    if (reports.isEmpty) {
      reports.add(
        Report(
          plantName: "Не найдено",
          imagePath: '',
        ),
      );
    }

    return reports;
  }



  List<double> softmax(List<double> logits) {
    // 1. Находим max для численной стабильности
    final double maxLogit = logits.reduce(max);

    // 2. Вычисляем exp(logit - maxLogit) для каждого элемента
    final exps = logits.map((x) => exp(x - maxLogit)).toList();

    // 3. Суммируем все exp
    final double sumExps = exps.reduce((a, b) => a + b);

    // 4. Делим каждый exp на сумму
    final List<double> probs = exps.map((x) => x / sumExps).toList();

    return probs;
  }






}