import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'processing_screen.dart';
import 'package:plant_analyzer/screens/report_history_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  /// Выбор изображения на Android
  Future<File?> _pickImageAndroid() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      return File(pickedFile.path);
    }
    return null;
  }

  /// Выбор изображения на Windows
  Future<File?> _pickImageWindows() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null && result.files.single.path != null) {
      return File(result.files.single.path!);
    }
    return null;
  }

  /// Универсальный метод для выбора изображения
  Future<File?> _pickImage() async {
    if (Platform.isAndroid) return _pickImageAndroid();
    if (Platform.isWindows) return _pickImageWindows();
    return null;
  }

/// Переход к истории отчетов
void _openHistory(BuildContext context) {
  Navigator.push(
    context,
    MaterialPageRoute(builder: (_) => ReportHistoryScreen()),
  );
}


  /// Съемка фото на Android (заглушка)
  void _takePhoto(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Съемка фото пока не реализована')),
    );
  }

  @override
  Widget build(BuildContext context) {
    bool isAndroid = !kIsWeb && Platform.isAndroid;
    bool isWindows = !kIsWeb && Platform.isWindows;

    return Scaffold(
      appBar: AppBar(title: const Text('PlantAnalyzer')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isAndroid)
              ElevatedButton(
                onPressed: () => _takePhoto(context),
                child: const Text('Сделать фото'),
              ),
            if (isAndroid || isWindows)
              ElevatedButton(
                onPressed: () async {
                  final file = await _pickImage();
                  if (file != null) {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => ProcessingScreen(imageFile: file),
                      ),
                    );
                  }
                },
                child: const Text('Выбрать фото с устройства'),
              ),
            ElevatedButton(
              onPressed: () => _openHistory(context),
              child: const Text('История отчетов'),
            ),
          ],
        ),
      ),
    );
  }
}
