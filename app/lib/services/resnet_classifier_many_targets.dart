import 'dart:typed_data';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';

class ResnetClassifierManyTargets {
  late Interpreter _interpreter;
  bool _initialized = false;

  /// Инициализация модели
  Future<void> init({
    required String modelAsset,
  }) async {
    if (_initialized) return;
    _interpreter = await Interpreter.fromAsset(modelAsset);
    _initialized = true;
  }

  /// Предобработка изображения: resize → center crop → normalize (CHW)
  List<List<List<double>>> preprocess(img.Image image) {
  const targetSize = 320;

  // resize
  int newWidth, newHeight;
  if (image.width < image.height) {
    newWidth = targetSize;
    newHeight = (targetSize * image.height / image.width).round();
  } else {
    newHeight = targetSize;
    newWidth = (targetSize * image.width / image.height).round();
  }

  // ресайз
  img.Image resized = img.copyResize(image, width: newWidth, height: newHeight);

  // центр-кроп
  int left = ((resized.width - targetSize) / 2).round();
  int top = ((resized.height - targetSize) / 2).round();

  img.Image cropped = img.copyCrop(
    resized,
    x: left,
    y: top,
    width: targetSize,
    height: targetSize,
  );


  // HWC
  List<List<List<double>>> hwc = List.generate(
    targetSize,
    (_) => List.generate(
      targetSize,
      (_) => List.filled(3, 0.0),
    ),
  );

  List<double> rValues = [];
  List<double> gValues = [];
  List<double> bValues = [];

  for (int y = 0; y < targetSize; y++) {
    for (int x = 0; x < targetSize; x++) {
      final pixel = cropped.getPixel(x, y); // Pixel
      final rnum = pixel.rNormalized;
      final gnum = pixel.gNormalized;
      final bnum = pixel.bNormalized;
      double r = rnum.toDouble();
      double g = gnum.toDouble();
      double b = bnum.toDouble();
      rValues.add(r);
      gValues.add(g);
      bValues.add(b);

      hwc[y][x][0] = r;
      hwc[y][x][1] = g;
      hwc[y][x][2] = b;
    }
  }



  // mean/std
  double mean(List<double> values) => values.reduce((a, b) => a + b) / values.length;
  double std(List<double> values, double m) =>
      sqrt(values.map((v) => (v - m) * (v - m)).reduce((a, b) => a + b) / values.length);

  const meanR = 0.485;
  const meanG = 0.456;
  const meanB = 0.406;

  const stdR = 0.229;
  const stdG = 0.224;
  const stdB = 0.225;

  // normalize
  for (int y = 0; y < targetSize; y++) {
    for (int x = 0; x < targetSize; x++) {
      hwc[y][x][0] = (hwc[y][x][0] - meanR) / stdR;
      hwc[y][x][1] = (hwc[y][x][1] - meanG) / stdG;
      hwc[y][x][2] = (hwc[y][x][2] - meanB) / stdB;
    }
  }

  return hwc;
}



  /// Softmax
  List<double> softmax(List<double> logits) {
    final double maxLogit = logits.reduce(max);
    final exps = logits.map((x) => exp(x - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sumExps).toList();
  }

  /// Классификация изображения
  Future<Map<String, List<double>>> classify(img.Image image) async {
  if (!_initialized) {
    throw Exception("Classifier not initialized!");
  }

  final hwc = preprocess(image); // [320,320,3]
  final input = [hwc];           // [1,320,320,3]

  // Подготавливаем все выходные тензоры
  final outputs = <int, Object>{};
  for (int i = 0; i <= 6; i++) {
    final shape = _interpreter.getOutputTensor(i).shape;
    outputs[i] = List.generate(shape[0], (_) => List.filled(shape[1], 0.0));
  }

  // Запуск модели
  _interpreter.runForMultipleInputs([input], outputs);

  // Названия выходов по порядку
  const outputNames = [
    'trunkHoles',
    'trunkCracks',
    'fruitingBodies',
    'overallCondition',
    'crownDamage',
    'trunkDamage',
    'trunkRot',
  ];

  // Формируем результаты с читаемыми ключами
  final Map<String, List<double>> results = {};
  for (int i = 0; i <= 6; i++) {
    final logits = (outputs[i] as List<List<double>>)[0];
    results[outputNames[i]] = softmax(logits);
  }
  debugPrint(results.toString());
  return results;
}

String _getLabel(List<double> probs, Map<int, String> labels) {
  final maxIndex = probs.indexWhere((p) => p == probs.reduce((a, b) => a > b ? a : b));
  return labels[maxIndex] ?? 'Не определено';
}

// 1. Полость (trunkHoles)
String getPropertyTrunkHoles(Map<String, List<double>> results) {
  const labels = {0: 'Нет', 1: 'Есть'};
  return _getLabel(results['trunkHoles'] ?? [], labels);
}

// 2. Трещины (trunkCracks)
String getPropertyTrunkCracks(Map<String, List<double>> results) {
  const labels = {0: 'Нет', 1: 'Есть'};
  return _getLabel(results['trunkCracks'] ?? [], labels);
}

// 3. Плоды или грибы (fruitingBodies)
String getPropertyFruitingBodies(Map<String, List<double>> results) {
  const labels = {0: 'Нет', 1: 'Есть'};
  return _getLabel(results['fruitingBodies'] ?? [], labels);
}

// 4. Общая оценка состояния (overallCondition)
String getPropertyOverallCondition(Map<String, List<double>> results) {
  const labels = {
    0: 'Не определено',
    1: 'Аварийное',
    2: 'Нездоровое',
    3: 'Нормальное',
    4: 'Опасное',
    5: 'Хорошее',
  };
  return _getLabel(results['overallCondition'] ?? [], labels);
}

// 5. Повреждение кроны (crownDamage)
String getPropertyCrownDamage(Map<String, List<double>> results) {
  const labels = {0: 'Нет', 1: 'Есть'};
  return _getLabel(results['crownDamage'] ?? [], labels);
}

// 6. Повреждение ствола (trunkDamage)
String getPropertyTrunkDamage(Map<String, List<double>> results) {
  const labels = {0: 'Нет', 1: 'Есть'};
  return _getLabel(results['trunkDamage'] ?? [], labels);
}

// 7. Гниль ствола (trunkRot)
String getPropertyTrunkRot(Map<String, List<double>> results) {
  const labels = {0: 'Нет', 1: 'Есть'};
  return _getLabel(results['trunkRot'] ?? [], labels);
}




  
}
