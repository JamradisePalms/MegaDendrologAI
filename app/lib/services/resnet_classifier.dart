import 'dart:typed_data';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ResnetClassifier {
  late Interpreter _interpreter;
  late List<String> _classNames;
  bool _initialized = false;

  /// Инициализация модели
  Future<void> init({
    required String modelAsset,
    required List<String> classNames,
  }) async {
    if (_initialized) return;
    _interpreter = await Interpreter.fromAsset(modelAsset);
    _classNames = classNames;
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
  img.Image resized = img.copyResize(image, width: newWidth, height: newHeight);

  // center crop
  int left = ((resized.width - targetSize) / 2).round();
  int top = ((resized.height - targetSize) / 2).round();
  img.Image cropped = img.copyCrop(resized, left, top, targetSize, targetSize);

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
      final pixel = cropped.getPixel(x, y);
      final r = img.getRed(pixel) / 255.0;
      final g = img.getGreen(pixel) / 255.0;
      final b = img.getBlue(pixel) / 255.0;

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

  final meanR = mean(rValues);
  final meanG = mean(gValues);
  final meanB = mean(bValues);

  final stdR = std(rValues, meanR);
  final stdG = std(gValues, meanG);
  final stdB = std(bValues, meanB);

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
  Future<Map<String, double>> classify(img.Image image) async {
    if (!_initialized) {
      throw Exception("Classifier not initialized!");
    }

    // preprocess -> [3, 320, 320]
    final hwc = preprocess(image);

    // Формируем вход [1, 3, 320, 320]
    final input = [hwc];

    // Выход: [1, num_classes]
    final outputShape = _interpreter.getOutputTensor(0).shape;
    final output = List.generate(1, (_) => List.filled(outputShape[1], 0.0));

    _interpreter.run(input, output);

    final logits = output[0].cast<double>();
    final probs = softmax(logits);

    // собираем в map
    final Map<String, double> results = {};
    for (int i = 0; i < _classNames.length; i++) {
      results[_classNames[i]] = probs[i];
    }

    return results;
  }

  /// Взять лучший класс
  Future<String> predictLabel(img.Image image) async {
    final results = await classify(image);
    final best = results.entries.reduce((a, b) => a.value > b.value ? a : b);
    return "${best.key} (${(best.value * 100).toStringAsFixed(1)}%)";
  }
}
