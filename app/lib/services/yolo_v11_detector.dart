import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';


class YoloV11Detector {
  Interpreter? _interpreter;
  final int inputSize = 640;
  final int anchors = 8400; // количество предложений (anchors)
  final List<String> classNames;
  final int numOutputs = 6; // [x, y, w, h, score, class_id]

  YoloV11Detector._(this.classNames);

  /// Factory: создаёт detector из asset (рекомендуется) или из произвольного пути.
  /// Если modelAsset != null, будет использован Interpreter.fromAsset,
  /// иначе будет загружен файл по modelPath через fromBuffer.
  static Future<YoloV11Detector> create({
    required List<String> classNames,
    String? modelAsset,
    String? modelPath,
    int threads = 4,
    bool tryUseGPU = false,
  }) async {
    final det = YoloV11Detector._(classNames);

    final options = InterpreterOptions()..threads = threads;

    // Временно закомментируем GPU delegate
    // if (tryUseGPU) {
    //   try {
    //     final gpu = GpuDelegateV2();
    //     options.addDelegate(gpu);
    //   } catch (e) {
    //     print('GPU delegate not available, falling back to CPU: $e');
    //   }
    // }

    if (modelAsset != null) {
      det._interpreter = await Interpreter.fromAsset(modelAsset, options: options);
      
    } else if (modelPath != null) {
      final bytes = File(modelPath).readAsBytesSync();
      det._interpreter = await Interpreter.fromBuffer(bytes, options: options);
    } else {
      throw ArgumentError('Either modelAsset or modelPath must be provided');
    }

    return det;
  }

  /// Основной метод: принимает ui.Image (например из Image.memory или из camera)
  /// и возвращает список Detection.
  Future<List<Detection>> detect(
    ui.Image image, {
    double confidenceThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    if (_interpreter == null) throw StateError('Interpreter is not initialized');
    debugPrint('before letterboxing');
    // 1) Letterbox resize + получить img.Image (RGB)
    final img.Image letterboxed = await _letterboxImage(image, inputSize, inputSize);
    debugPrint('after letterboxing in detect');
    // 2) Подготовить input tensor: [1,640,640,3], float32 normalized 0..1
    final input = _imageToFloatInput(letterboxed);

    // Создаем правильную 4D структуру [batch, height, width, channels]
    final inputTensor = List.generate(1, (batch) {
      return List.generate(inputSize, (y) {
        return List.generate(inputSize, (x) {
          final idx = (y * inputSize + x) * 3;
          return [input[idx], input[idx + 1], input[idx + 2]];
        });
      });
    });

    // 3) Подготовка output: [1, 6, anchors]
    final output = List.generate(1, (_) => List.generate(numOutputs, (_) => List.filled(anchors, 0.0)));

    // 4) Run
    _interpreter!.run(inputTensor, output);

    // 5) Постобработка и NMS
    final detections = _postprocessOutput(
      output,
      originalWidth: image.width,
      originalHeight: image.height,
      confThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
    );

    return detections;
  }

  /// Letterbox: ресайз с сохранением пропорций + заполнение серым (128)
  Future<img.Image> _letterboxImage(ui.Image srcImage, int targetW, int targetH) async {
    // получить RGBA байты
  final byteData = await srcImage.toByteData(format: ui.ImageByteFormat.rawRgba);
  
  if (byteData == null) throw Exception('Failed to get image bytes');

  final ByteBuffer buffer = byteData.buffer; // ByteBuffer из dart:typed_data

  // Конвертация в package:image Image (формат RGBA)
  final img.Image original = img.Image.fromBytes(
    width: srcImage.width,
    height: srcImage.height,
    bytes: buffer,
    bytesOffset: 0,
    format: img.Format.uint8,
    numChannels: 4, // RGBA
  );
  

    // Рассчитать scale
    final double scale = <double>[targetW / original.width, targetH / original.height].reduce((a, b) => a < b ? a : b);
    final int newW = (original.width * scale).round();
    final int newH = (original.height * scale).round();

    // Resize (bilinear)
    final img.Image resized = img.copyResize(
      original,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.linear,
    );
    
    /// Создать фон и вставить по центру

    final bgColor = img.ColorUint8.rgba(128, 128, 128, 255);

    img.Image? canvas;

    
    canvas = img.copyExpandCanvas(
      resized,
      newWidth: targetW,
      newHeight: targetH,
      position: img.ExpandCanvasPosition.center,
      backgroundColor: bgColor,
    );

    // fallback — если не удалось, создаём вручную серый холст
    canvas = img.Image(width: targetW, height: targetH);
    img.fill(canvas!, color: bgColor);
    final dx = ((targetW - resized.width) / 2).round();
    final dy = ((targetH - resized.height) / 2).round();
    img.compositeImage(canvas!, resized, dstX: dx, dstY: dy);
    

    return canvas;



  }

  /// Конвертация img.Image -> Float32List normalized 0..1
  Float32List _imageToFloatInput(img.Image image) {
    final int w = image.width;
    final int h = image.height;
    final Float32List buffer = Float32List(1 * inputSize * inputSize * 3);

    int outIdx = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final img.Pixel pixel = image.getPixel(x, y);

        // берём нормализованные каналы сразу (0..1)
        buffer[outIdx++] = pixel.rNormalized.toDouble();
        buffer[outIdx++] = pixel.gNormalized.toDouble();
        buffer[outIdx++] = pixel.bNormalized.toDouble();
      }
    }

    return buffer;
  }


  /// Постобработка выхода [1,6,anchors] -> список Detection
  List<Detection> _postprocessOutput(
    List<List<List<double>>> output, {
    required int originalWidth,
    required int originalHeight,
    required double confThreshold,
    required double iouThreshold,
  }) {
    // output[0][channel][anchor]
    final List<BoundingBox> candidates = [];

    for (int a = 0; a < anchors; a++) {
      final double xc = output[0][0][a];
      final double yc = output[0][1][a];
      final double w = output[0][2][a];
      final double h = output[0][3][a];
      final double score = output[0][4][a];
      final double clsRaw = output[0][5][a];

      if (score < confThreshold) continue;

      int classId = clsRaw.round();
      if (classId < 0) classId = 0;
      if (classId >= classNames.length) classId = classNames.length - 1;

      final double left = xc - w / 2.0;
      final double top = yc - h / 2.0;

      candidates.add(BoundingBox(
        left: left,
        top: top,
        width: w,
        height: h,
        confidence: score,
        classId: classId,
      ));
    }

    // NMS (по классам)
    final List<BoundingBox> kept = _nms(candidates, iouThreshold);

    // Преобразуем координаты из letterboxed (нормализованные по inputSize) обратно в координаты оригинала (пиксели)
    // Нужно вычислить scale и padding как в letterbox
    final double scale = <double>[inputSize / originalWidth, inputSize / originalHeight].reduce((a, b) => a < b ? a : b);
    final double newW = originalWidth * scale;
    final double newH = originalHeight * scale;
    final double padW = (inputSize - newW) / 2.0;
    final double padH = (inputSize - newH) / 2.0;

    final List<Detection> detections = [];

    for (final box in kept) {
      // box coords сейчас в normalized относительно inputSize (0..inputSize)
      // Если модель уже даёт координаты в 0..1, возможно потребуется умножение на inputSize.
      // Предполагаем, что xc,yc,w,h у модели в координатах относительно inputSize (т.е. 0..inputSize).
      // Если модель возвращает 0..1, тогда нужно умножить на inputSize. Проверяй на своей модели.
      // Здесь поддержим оба варианта: если значения <= 1.0 — считаем нормализованными (0..1).
      double left = box.left;
      double top = box.top;
      double width = box.width;
      double height = box.height;

      // Если координаты в 0..1, конвертируем в pixels on input
      if (left <= 1.01 && top <= 1.01 && width <= 1.01 && height <= 1.01) {
        left = left * inputSize;
        top = top * inputSize;
        width = width * inputSize;
        height = height * inputSize;
      }

      // Уберём паддинг и переведём в координаты оригинала
      final double x1 = ((left - padW) / scale).clamp(0.0, originalWidth.toDouble());
      final double y1 = ((top - padH) / scale).clamp(0.0, originalHeight.toDouble());
      final double wOut = (width / scale).clamp(0.0, originalWidth.toDouble() - x1);
      final double hOut = (height / scale).clamp(0.0, originalHeight.toDouble() - y1);

      detections.add(Detection(
        className: classNames[box.classId],
        confidence: box.confidence,
        x: x1,
        y: y1,
        width: wOut,
        height: hOut,
      ));
    }

    return detections;
  }

  /// NMS по классам
  List<BoundingBox> _nms(List<BoundingBox> boxes, double iouThreshold) {
    if (boxes.isEmpty) return [];

    // Группируем по classId
    final Map<int, List<BoundingBox>> byClass = {};
    for (final b in boxes) {
      byClass.putIfAbsent(b.classId, () => []).add(b);
    }

    final List<BoundingBox> result = [];

    for (final entry in byClass.entries) {
      final List<BoundingBox> list = List.from(entry.value);
      list.sort((a, b) => b.confidence.compareTo(a.confidence));

      final List<bool> suppressed = List<bool>.filled(list.length, false);

      for (int i = 0; i < list.length; i++) {
        if (suppressed[i]) continue;
        final current = list[i];
        result.add(current);

        for (int j = i + 1; j < list.length; j++) {
          if (suppressed[j]) continue;
          final other = list[j];
          final double iou = _computeIoU(current, other);
          if (iou > iouThreshold) suppressed[j] = true;
        }
      }
    }

    return result;
  }

  double _computeIoU(BoundingBox a, BoundingBox b) {
    final double left = a.left > b.left ? a.left : b.left;
    final double top = a.top > b.top ? a.top : b.top;
    final double right = (a.left + a.width) < (b.left + b.width) ? (a.left + a.width) : (b.left + b.width);
    final double bottom = (a.top + a.height) < (b.top + b.height) ? (a.top + a.height) : (b.top + b.height);

    if (right <= left || bottom <= top) return 0.0;

    final interArea = (right - left) * (bottom - top);
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    return interArea / unionArea;
  }

  /// Закрыть интерпретер (освобождение ресурсов)
  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}

/// Вспомогательные классы
class BoundingBox {
  double left;
  double top;
  double width;
  double height;
  final double confidence;
  final int classId;

  BoundingBox({
    required this.left,
    required this.top,
    required this.width,
    required this.height,
    required this.confidence,
    required this.classId,
  });
}

class Detection {
  final String className;
  final double confidence;
  final double x; // left (px)
  final double y; // top (px)
  final double width; // px
  final double height; // px

  Detection({
    required this.className,
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}
