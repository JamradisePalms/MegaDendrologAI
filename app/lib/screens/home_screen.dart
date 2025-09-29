import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'processing_screen.dart';
import 'package:plant_analyzer/screens/report_history_screen.dart';
import 'crop_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _isProcessing = false;

  /// Выбор изображения на Android (галерея)
  Future<File?> _pickImageAndroid() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) return File(pickedFile.path);
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

  /// Съемка фото на Android — теперь реализовано
  Future<void> _takePhoto() async {
    if (kIsWeb || !Platform.isAndroid) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Съемка фото поддерживается только на Android в этом экране')),
      );
      return;
    }

    try {
      setState(() => _isProcessing = true);

      final XFile? picked = await _picker.pickImage(
        source: ImageSource.camera,
        preferredCameraDevice: CameraDevice.rear,
        imageQuality: 85, // компрессия, 0-100 (опционально)
      );

      if (picked == null) {
        // пользователь отменил
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Съемка отменена')),
          );
        }
        return;
      }

      final File imageFile = File(picked.path);

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => CropScreen(imageFile: imageFile),
        ),
      );

    } catch (e) {
      debugPrint('Ошибка при съемке: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка при съемке: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
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
                onPressed: _isProcessing ? null : _takePhoto,
                child: _isProcessing
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Сделать фото'),
              ),
            if (isAndroid || isWindows)
              ElevatedButton(
                onPressed: _isProcessing
                    ? null
                    : () async {
                        setState(() => _isProcessing = true);
                        try {
                          final file = await _pickImage();
                          if (file != null && mounted) {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (_) => ProcessingScreen(imageFile: file),
                              ),
                            );
                          } else {
                            if (mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(content: Text('Файл не выбран')),
                              );
                            }
                          }
                        } finally {
                          if (mounted) setState(() => _isProcessing = false);
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
