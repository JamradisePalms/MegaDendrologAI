import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../models/report.dart';
import '../services/api_service.dart';
import '../services/local_analysis.dart';
import 'report_screen.dart';
import '../services/report_service.dart';

class ProcessingScreen extends StatefulWidget {
  final File imageFile;

  const ProcessingScreen({super.key, required this.imageFile});

  @override
  _ProcessingScreenState createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  final ApiService _apiService = ApiService();
  final LocalAnalysis _localAnalysis = LocalAnalysis();

  @override
  void initState() {
    super.initState();
    _analyze();
  }
  Future<bool> _hasInternet() async {
    try {
      final result = await InternetAddress.lookup('google.com');
      return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
    } catch (_) {
      return false;
    }
  }
  Future<void> _analyze() async {
    try {
      // 1. Сохраняем копию файла в постоянное хранилище
      final docsDir = await getApplicationDocumentsDirectory();
      final fileName = 'report_${DateTime.now().millisecondsSinceEpoch}${p.extension(widget.imageFile.path)}';
      final savedImage = await widget.imageFile.copy(p.join(docsDir.path, fileName));

      // 2. Проверяем интернет
      final internetAvailable = await _hasInternet();

      Report report;
      if (internetAvailable) {
        debugPrint('Есть интернет → используем ApiService для файла: ${savedImage.path}');
        report = await _apiService.analyzeImage(savedImage);
      } else {
        debugPrint('Нет интернета → используем LocalAnalysis для файла: ${savedImage.path}');
        report = await _localAnalysis.analyzeImage(savedImage);
      }


      // 3. Создаём новый Report с правильным путём (без copyWith)
      final reportWithPath = Report(
        plantName: report.plantName,
        probability: report.probability,
        species: report.species,
        trunkRot: report.trunkRot,
        trunkHoles: report.trunkHoles,
        trunkCracks: report.trunkCracks,
        trunkDamage: report.trunkDamage,
        crownDamage: report.crownDamage,
        fruitingBodies: report.fruitingBodies,
        diseases: report.diseases,
        dryBranchPercentage: report.dryBranchPercentage,
        additionalInfo: report.additionalInfo,
        imagePath: savedImage.path,
      );

      // 4. Сохраняем в кэш
      await ReportService().saveReport(reportWithPath);

      // 5. Переход на экран отчета
      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ReportScreen(report: reportWithPath),
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
