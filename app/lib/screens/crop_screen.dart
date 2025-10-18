import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:crop_your_image/crop_your_image.dart';
import 'processing_screen.dart';

class CropScreen extends StatefulWidget {
  final File imageFile;
  const CropScreen({super.key, required this.imageFile, this.geoData});
  final String? geoData;

  @override
  State<CropScreen> createState() => _CropScreenState();
}

class _CropScreenState extends State<CropScreen> {
  final CropController _controller = CropController();
  bool _isCropping = false;
  Uint8List? _imageData;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final bytes = await widget.imageFile.readAsBytes();
    setState(() => _imageData = bytes);
  }

  void _cropImage() {
    if (_isCropping || _imageData == null) return;
    setState(() => _isCropping = true);
    _controller.crop();
  }

  void _autoDetect() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ProcessingScreen(imageFile: widget.imageFile),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Выделите растение')),
      body: _imageData == null
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: Crop(
                    image: _imageData!,
                    controller: _controller,
                    aspectRatio: null,
                    interactive: false,
                    fixCropRect: false,
                    withCircleUi: false,
                    maskColor: Colors.black.withAlpha(180),
                    baseColor: Colors.white,
                    progressIndicator: const CircularProgressIndicator(),

                    // 🔹 показываем всё изображение изначально
                    initialRectBuilder: InitialRectBuilder.withBuilder((viewportRect, imageRect) {
                      // Задаём коэффициент уменьшения кропа (например, 70% от изображения)
                      const double factor = 0.7;

                      // Рассчитываем масштаб, чтобы всё изображение помещалось
                      final scaleX = viewportRect.width / imageRect.width;
                      final scaleY = viewportRect.height / imageRect.height;
                      final scale = scaleX < scaleY ? scaleX : scaleY; // минимальный масштаб

                      // Размер кропа меньше изображения
                      final width = imageRect.width * scale * factor;
                      final height = imageRect.height * scale * factor;

                      // Центрируем кроп-рамку
                      final left = viewportRect.left + (viewportRect.width - width) / 2;
                      final top = viewportRect.top + (viewportRect.height - height) / 2;

                      return Rect.fromLTWH(left, top, width, height);
                    }),




                    onCropped: (result) {
                      setState(() => _isCropping = false);
                      switch (result) {
                        case CropSuccess(:final croppedImage):
                          final tempFile =
                              File('${Directory.systemTemp.path}/cropped.png');
                          tempFile.writeAsBytesSync(croppedImage);
                          if (!mounted) return;
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) =>
                                  ProcessingScreen(imageFile: tempFile, isCroppedByUser: true, geoData: widget.geoData),
                            ),
                          );
                        case CropFailure(:final cause):
                          debugPrint('Ошибка кропа: $cause');
                      }
                    },
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton(
                        onPressed: _isCropping ? null : _cropImage,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          minimumSize: const Size(100, 36), // уменьшенный размер кнопки
                          textStyle: const TextStyle(fontSize: 14),
                        ),
                        child: _isCropping
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Text('Растение выделено'),
                      ),
                      ElevatedButton(
                        onPressed: _autoDetect,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          minimumSize: const Size(100, 36),
                          textStyle: const TextStyle(fontSize: 14),
                        ),
                        child: const Text('Автовыделение'),
                      ),
                    ],
                  ),
                ),

              ],
            ),
    );
  }
}
