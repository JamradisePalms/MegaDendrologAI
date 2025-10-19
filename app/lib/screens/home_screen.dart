import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'processing_screen.dart';
import 'package:plant_analyzer/screens/report_history_screen.dart';
import 'crop_screen.dart';
import 'package:geolocator/geolocator.dart';
import '../services/analysis_queue_dao.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  final _queueDao = AnalysisQueueDao();

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

  Future<void> _pickMultipleImagesAndQueue() async {
    try {
      setState(() => _isProcessing = true);

      List<File> imageFiles = [];

      if (Platform.isAndroid) {
        final List<XFile>? pickedFiles = await _picker.pickMultiImage(imageQuality: 85);
        if (pickedFiles != null && pickedFiles.isNotEmpty) {
          imageFiles = pickedFiles.map((x) => File(x.path)).toList();
        }
      } else if (Platform.isWindows) {
        final result = await FilePicker.platform.pickFiles(
          allowMultiple: true,
          type: FileType.image,
        );
        if (result != null && result.files.isNotEmpty) {
          imageFiles = result.paths.whereType<String>().map((p) => File(p)).toList();
        }
      }

      if (imageFiles.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Файлы не выбраны')),
          );
        }
        return;
      }

      // 🚀 Добавляем все выбранные фото в очередь
      for (final file in imageFiles) {
        await _queueDao.addTask(
          file.path,
          0, // reportId пока можно ставить 0, если еще не создан
          onlyOnWifi: false,
        );
      }

      await _queueDao.debugPrintQueue();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Добавлено в очередь: ${imageFiles.length} файлов. Результаты появятся в истории отчётов.')),
        );
      }
    } catch (e) {
      debugPrint('Ошибка при загрузке массива фото: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }
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
        imageQuality: 85,
      );

      if (picked == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Съемка отменена')),
          );
        }
        return;
      }

      final File imageFile = File(picked.path);

      // ✅ Получение геопозиции устройства
      String? geoData;
      final position = await _getCurrentPosition();
      debugPrint('position: $position');
      if (position != null) {
        geoData = '${position.latitude}, ${position.longitude}';
        debugPrint('geoData: $geoData');
      }

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ProcessingScreen(
            imageFile: imageFile,
            geoData: geoData, // передаем координаты
          ),
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
  


  /// Получение текущей позиции устройства через Geolocator
  Future<Position?> _getCurrentPosition() async {
    bool serviceEnabled;
    LocationPermission permission;

    // Проверка, включена ли служба геолокации
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      debugPrint('Служба геолокации выключена.');
      return null;
    }

    // Проверка и запрос разрешений
    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        debugPrint('Разрешение на геолокацию отклонено');
        return null;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      debugPrint('Разрешение на геолокацию отклонено навсегда');
      return null;
    }

    // Получение позиции
    LocationSettings locationSettings;
    if (Platform.isAndroid) {
      locationSettings = AndroidSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 0,
      );
    } else if (Platform.isIOS) {
      locationSettings = AppleSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 0,
      );
    } else {
      locationSettings = const LocationSettings(
        accuracy: LocationAccuracy.high,
      );
    }

    return await Geolocator.getCurrentPosition(locationSettings: locationSettings);
  }



@override
Widget build(BuildContext context) {
  bool isAndroid = !kIsWeb && Platform.isAndroid;
  bool isWindows = !kIsWeb && Platform.isWindows;

  const double iconSize = 36;
  const double buttonWidth = 90;
  const double buttonHeight = 90;

  Widget buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
  }) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: buttonWidth,
          height: buttonHeight,
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              padding: const EdgeInsets.all(8),
              backgroundColor: Theme.of(context).colorScheme.primary,
            ),
            onPressed: onPressed,
            child: Icon(icon, size: iconSize, color: Colors.white),
          ),
        ),
        const SizedBox(height: 6),
        SizedBox(
          width: buttonWidth + 15,
          child: Text(
            label,
            textAlign: TextAlign.center,
            softWrap: true,
            maxLines: 4,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontSize: 14),
          ),
        ),
      ],
    );
  }

  return Scaffold(
    appBar: AppBar(
      leading: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Image.asset(
          'assets/images/icon_white.png',
          width: 24,
          height: 24,
          fit: BoxFit.contain,
        ),
      ),
      title: const Text('PlantGuard'),
    ),

    // 📍 Центрируем ряд кнопок по вертикали
    body: Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          buildActionButton(
            icon: Icons.image,
            label: 'Выбрать фото для анализа',
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
                            builder: (_) => CropScreen(imageFile: file),
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
          ),
          const SizedBox(width: 20),
          buildActionButton(
            icon: Icons.collections,
            label: 'Загрузить несколько фото в очередь',
            onPressed: _isProcessing ? null : _pickMultipleImagesAndQueue,
          ),
          const SizedBox(width: 20),
          buildActionButton(
            icon: Icons.history,
            label: 'История отчетов',
            onPressed: () => _openHistory(context),
          ),
        ],
      ),
    ),

    // 📸 Нижняя круглая кнопка
    floatingActionButton: isAndroid
        ? FloatingActionButton(
            backgroundColor: Theme.of(context).colorScheme.primary,
            onPressed: _isProcessing ? null : _takePhoto,
            child: _isProcessing
                ? const SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(
                      color: Colors.white,
                      strokeWidth: 2.5,
                    ),
                  )
                : const Icon(Icons.camera_alt, color: Colors.white, size: 32),
          )
        : null,
    floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
  );
}





}
