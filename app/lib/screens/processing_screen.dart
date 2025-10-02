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

class ProcessingScreen extends StatefulWidget {
  final File imageFile;

  const ProcessingScreen({super.key, required this.imageFile});

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
    final fileName = 'report_${DateTime.now().millisecondsSinceEpoch}${p.extension(widget.imageFile.path)}';
    final savedImage = await widget.imageFile.copy(p.join(docsDir.path, fileName));

    // 2. Проверяем интернет
    final internetAvailable = await ConnectivityService.hasInternet();

    List<Report> reports = [];
    if (internetAvailable) {
      debugPrint('Есть интернет → используем ApiService для файла: ${savedImage.path}');
      //reports = await _localAnalysis.analyzeImage(savedImage);
      reports = await _apiService.analyzeImage(savedImage);
    } else {
      debugPrint('Нет интернета → используем LocalAnalysis для файла: ${savedImage.path}');
      reports = await _localAnalysis.analyzeImage(savedImage);
    }

    // 3. Сохраняем в кэш все отчёты
    /*final birchReport = Report(
      id: 778, // пока новый отчет, id будет присвоен БД
      plantName: 'Берёза, 2 октября 2025',
      probability: 92.5,
      species: 'Берёза',
      trunkRot: 'Нет',
      trunkHoles: 'Нет',
      trunkCracks: 'Мелкие',
      trunkDamage: 'Нет',
      crownDamage: 'Нет',
      fruitingBodies: 'Нет',
      diseases: 'Нет',
      dryBranchPercentage: 5.0,
      additionalInfo: 'Состояние хорошее, без видимых заболеваний',
      overallCondition: 'Здоровое дерево',
      imagePath: 'assets/images/20200724_115413.jpg',
      imageUrl: null,
      analyzedAt: DateTime.now().toIso8601String(),
    );*/
    
    final reportService = ReportService();

    //final reportId = await reportService.saveReport(birchReport);
    for (final r in reports) {
      final reportId = await reportService.saveReport(r);
      if (!internetAvailable) {
        await queueDao.addTask(savedImage.path, reportId);
      }
    }

    if (!internetAvailable) {
      await AnalysisQueueDao().debugPrintQueue();
    }

    // 4. Переход на экран отчёта
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => ReportScreen(reports: reports), // 👈 лучше сразу список
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
