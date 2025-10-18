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
import '../services/resnet_classifier_many_targets.dart';
import 'package:flutter/foundation.dart';
import 'package:intl/intl.dart'; // для форматирования даты
import 'dart:typed_data';



class LocalAnalysis {
  late YoloV11Detector _detector;
  late ResnetClassifier _resnetClassifier; // поле класса
  late ResnetClassifierManyTargets _resnetClassifierManyTargets;
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
    _resnetClassifierManyTargets = ResnetClassifierManyTargets();
    await _resnetClassifierManyTargets.init(
      modelAsset: 'assets/models/resnet_many_targets2.tflite',
    );

    _initialized = true;
  }

  Future<List<Report>> analyzeImage(
    File imageFile, {
    bool isCroppedByUser = false,
  }) async {
    if (!_initialized) {
      await init();
    }
    debugPrint('initialised');

    final bytes = await imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final ui.Image uiImage = frame.image;

    final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) throw Exception('Ошибка конвертации изображения');

    final ByteBuffer buffer = byteData.buffer;

    final img.Image baseImage = img.Image.fromBytes(
      width: uiImage.width,
      height: uiImage.height,
      bytes: buffer,
      bytesOffset: 0,
      format: img.Format.uint8,
      numChannels: 4, // RGBA
    );

    final Directory dir = await getApplicationDocumentsDirectory();
    final appDir = Directory('${dir.path}/PlantGuard');
    if (!await appDir.exists()) {
      await appDir.create(recursive: true);
    }

    final List<Report> reports = [];

    List<dynamic> detections = [];
    debugPrint(isCroppedByUser.toString());
    // --- если пользователь сам обрезал изображение — пропускаем детекцию ---
    if (!isCroppedByUser) {
      detections = await _detector.detect(uiImage);
      debugPrint('DEBUG: detections count=${detections.length}');
    } else {
      debugPrint('Пользователь обрезал изображение — детекция пропущена');
    }

    // Если есть детекции — обрабатываем каждую
    if (detections.isNotEmpty) {
      for (int i = 0; i < detections.length; i++) {
        final det = detections[i];

        final int x = det.x.round().clamp(0, baseImage.width - 1);
        final int y = det.y.round().clamp(0, baseImage.height - 1);
        final int w = det.width.round().clamp(1, baseImage.width - x);
        final int h = det.height.round().clamp(1, baseImage.height - y);

        final img.Image cropped =
            img.copyCrop(baseImage, x: x, y: y, width: w, height: h);

        await _processAndSaveReport(cropped, appDir, det.confidence, reports);
      }
    } else if (isCroppedByUser){
      // --- если пользователь сам кропнул ---
      await _processAndSaveReport(baseImage, appDir, 1.0, reports);
    }
    else{
      return reports;
    } 

    return reports;
  }

  // Вынесена логика анализа и сохранения одного растения
  Future<void> _processAndSaveReport(
    img.Image image,
    Directory appDir,
    double confidence,
    List<Report> reports,
  ) async {
    String plantName = 'Не определено';
    String top3species = 'Не определено';

    top3species = await _resnetClassifier.getTop3(image);
    plantName = await _resnetClassifier.getLabel(image);

    final classificationMap =
        await _resnetClassifierManyTargets.classify(image);

    final trunkHoles =
        _resnetClassifierManyTargets.getPropertyTrunkHoles(classificationMap);
    final trunkCracks =
        _resnetClassifierManyTargets.getPropertyTrunkCracks(classificationMap);
    final fruitingBodies =
        _resnetClassifierManyTargets.getPropertyFruitingBodies(classificationMap);
    final overallCondition =
        _resnetClassifierManyTargets.getPropertyOverallCondition(classificationMap);
    final crownDamage =
        _resnetClassifierManyTargets.getPropertyCrownDamage(classificationMap);
    final trunkDamage =
        _resnetClassifierManyTargets.getPropertyTrunkDamage(classificationMap);
    final trunkRot =
        _resnetClassifierManyTargets.getPropertyTrunkRot(classificationMap);

    final String outPath =
        "${appDir.path}/plant_${DateTime.now().millisecondsSinceEpoch}.png";
    final File outFile = File(outPath);
    await outFile.writeAsBytes(img.encodePng(image));

    final now = DateTime.now();
    final formattedDate = DateFormat('yyyy-MM-dd HH:mm').format(now);

    reports.add(
      Report(
        plantName: "$plantName $formattedDate",
        probability: double.parse((confidence * 100).toStringAsFixed(2)),
        species: top3species,
        trunkHoles: trunkHoles,
        trunkCracks: trunkCracks,
        fruitingBodies: fruitingBodies,
        overallCondition: overallCondition,
        crownDamage: crownDamage,
        trunkDamage: trunkDamage,
        trunkRot: trunkRot,
        imagePath: outFile.path,
        analyzedAt: formattedDate,
        isVerified: false,
      ),
    );
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