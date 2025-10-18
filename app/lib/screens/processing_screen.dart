import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../models/report.dart';
import '../services/api_service.dart';
import '../services/local_analysis.dart';
import 'report_screen.dart';
import '../services/report_service.dart';
import '../services/connectivity_service.dart';
import '../services/analysis_queue_dao.dart';
import 'crop_screen.dart';

class ProcessingScreen extends StatefulWidget {
  final File imageFile;
  final bool isCroppedByUser;
  final String? geoData; 

  const ProcessingScreen({
    super.key,
    required this.imageFile,
    this.isCroppedByUser = false,
    this.geoData, 
  });

  @override
  _ProcessingScreenState createState() => _ProcessingScreenState();
}


class _ProcessingScreenState extends State<ProcessingScreen> {
  final ApiService _apiService = ApiService();
  final LocalAnalysis _localAnalysis = LocalAnalysis();
  final queueDao = AnalysisQueueDao();

  @override
  void initState() {
    super.initState();
    _analyze();
  }
  Future<void> _analyze() async {
    try {
      // 1. Сохраняем копию файла в постоянное хранилище
      final docsDir = await getApplicationDocumentsDirectory();
      final fileName =
          'report_${DateTime.now().millisecondsSinceEpoch}${p.extension(widget.imageFile.path)}';
      final savedImage = await widget.imageFile.copy(p.join(docsDir.path, fileName));

      // 2. Проверяем интернет
      final internetAvailable = await ConnectivityService.hasInternet();

      List<Report> reports = [];
      if (internetAvailable) {
        debugPrint('Есть интернет → используем ApiService для файла: ${savedImage.path}');
        reports = await _apiService.analyzeImage(savedImage, isCroppedByUser: widget.isCroppedByUser, geoData: widget.geoData);
        //reports = await _localAnalysis.analyzeImage(savedImage);
      } else {
        debugPrint('Нет интернета → используем LocalAnalysis для файла: ${savedImage.path}');
        debugPrint(widget.isCroppedByUser.toString());
        reports = await _localAnalysis.analyzeImage(
          savedImage,
          isCroppedByUser: widget.isCroppedByUser,
          geoData: widget.geoData
        );

      }

      // 3. Проверяем, есть ли отчёты
      if (reports.isEmpty) {
      if (!mounted) return;
      await showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Результат анализа'),
          content: const Text('К сожалению растения не найдены на изображении - попробуйте выделить растение сами'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context), // закрываем диалог
              child: const Text('OK'),
            ),
          ],
        ),
      );

      // После закрытия диалога — переход на CropScreen
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => CropScreen(imageFile: savedImage, geoData: widget.geoData),
          ),
        );
      }
      return; // прекращаем дальнейшую обработку
    }

      // 4. Сохраняем в кэш все отчёты
      final reportService = ReportService();
      for (final r in reports) {
        debugPrint('Сохраняем отчёт: ${r.debugString()}');
        final reportId = await reportService.saveReport(r);
        if (!internetAvailable) {
          await queueDao.addTask(savedImage.path, reportId);
        }
      }

      if (!internetAvailable) {
        await AnalysisQueueDao().debugPrintQueue();
      }

      // 5. Переход на экран отчёта
      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ReportScreen(reports: reports),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка анализа: $e')),
      );
      Navigator.pop(context);
    }
  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Анализ изображения')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            const Text('Идет анализ изображения...'),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Отмена'),
            ),
          ],
        ),
      ),
    );
  }
}
