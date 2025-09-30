import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_cropper/image_cropper.dart';
import 'processing_screen.dart';

class CropScreen extends StatefulWidget {
  final File imageFile;
  const CropScreen({super.key, required this.imageFile});

  @override
  State<CropScreen> createState() => _CropScreenState();
}

class _CropScreenState extends State<CropScreen> {
  bool _isCropping = false;

  Future<void> _cropImage() async {
    if (_isCropping) return; // защита от двойного нажатия
    setState(() => _isCropping = true);

    try {
      final croppedFile = await ImageCropper().cropImage(
        sourcePath: widget.imageFile.path,
        compressQuality: 85,
        uiSettings: [
          AndroidUiSettings(
            toolbarTitle: 'Выделите растение',
            toolbarColor: Colors.green,
            toolbarWidgetColor: Colors.white,
            lockAspectRatio: false,
          ),
          IOSUiSettings(
            title: 'Выделите растение',
            aspectRatioLockEnabled: true,
          ),
        ],
      );

      if (croppedFile != null && mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ProcessingScreen(imageFile: File(croppedFile.path)),
          ),
        );
      } else {
        if (mounted) Navigator.pop(context);
      }
    } catch (e) {
      debugPrint('Ошибка кропа: $e');
      if (mounted) Navigator.pop(context);
    } finally {
      if (mounted) setState(() => _isCropping = false);
    }
  }

  /// Автодетекция — сразу отправляем фото на processing
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
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.file(widget.imageFile),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _cropImage,
              child: _isCropping
                  ? const CircularProgressIndicator()
                  : const Text('Выделить растение'),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              onPressed: _autoDetect,
              //style: ElevatedButton.styleFrom(Theme.of(context).colorScheme.primary),
              child: const Text('Автодетекция'),
            ),
          ],
        ),
      ),
    );
  }
}
